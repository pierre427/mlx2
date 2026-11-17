# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
# Scope and earlier grants: provenance/LICENSE-calculator-scope.json
# Terms: provenance/LICENSE-calculator.txt
"""Bounded exact decimal arithmetic on Metal; CPU builds IR and formats GPU digits only.

Original implementation. No CPU Fraction/eval in the GPU path. A GEMM candidate uses
FP32 digit convolution (all terms are small exact integers), then GPU carry propagation.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache

MAX_DIGITS = 60
MAX_INSTRUCTIONS = 256


@dataclass(frozen=True)
class Number:
    op: str
    args: tuple
    _calculator_ir = True

    @classmethod
    def literal(cls, text):
        return cls('literal', (text,))

    def __add__(self, x): return Number('+', (self, x))
    def __sub__(self, x): return Number('-', (self, x))
    def __mul__(self, x): return Number('*', (self, x))
    def __truediv__(self, x): return Number('/', (self, x))
    def __pow__(self, x): return Number('^', (self, x))
    def __neg__(self): return Number('neg', (self,))


def program(expr):
    root = expr.get('value') if expr['op'] == 'expr' else Number(expr['op'], (Number.literal(expr['a']), Number.literal(expr['b'])))
    if not isinstance(root, Number):
        raise ValueError('GPU evaluator requires deferred arithmetic IR')
    ops=[]; literals=[]; scales=[]
    def visit(n):
        for a in n.args if n.op!='literal' else ():
            visit(a)
        if len(ops)>=MAX_INSTRUCTIONS: raise ValueError('gpu_instruction_bound')
        if n.op=='literal':
            s=n.args[0].replace(',','');parts=s.split('.')
            if len(parts)>2 or any(not p.isascii() or not p.isdigit() for p in parts): raise ValueError('gpu_literal')
            digits=''.join(parts).lstrip('0') or '0';scale=len(parts[1]) if len(parts)==2 else 0
            if len(digits)>MAX_DIGITS or scale>MAX_DIGITS:raise ValueError('gpu_digit_bound')
            ops.append(0);literals.append([int(c) for c in digits[::-1]]+[0]*(128-len(digits)));scales.append(scale)
        else:
            ops.append({'+':1,'-':2,'*':3,'/':4,'^':5,'neg':6}[n.op]);literals.append([0]*128);scales.append(0)
    visit(root)
    return root,ops,literals,scales


_HEADER = r'''
#define CALC_NOUNROLL _Pragma("clang loop unroll(disable)")
constant int ND=128;
struct Big { uchar d[128]; };
struct Rat { Big n; Big den; int sign; };
inline Big zero_b(){Big a;CALC_NOUNROLL for(int i=0;i<ND;i++)a.d[i]=0;return a;}
inline Big one_b(){Big a=zero_b();a.d[0]=1;return a;}
inline bool iszero(Big a){CALC_NOUNROLL for(int i=0;i<ND;i++)if(a.d[i])return false;return true;}
inline int width(Big a){CALC_NOUNROLL for(int i=ND-1;i>=0;i--)if(a.d[i])return i+1;return 1;}
inline int cmp(Big a,Big b){CALC_NOUNROLL for(int i=ND-1;i>=0;i--){if(a.d[i]>b.d[i])return 1;if(a.d[i]<b.d[i])return -1;}return 0;}
inline Big sub_b(Big a,Big b){int borrow=0;CALC_NOUNROLL for(int i=0;i<ND;i++){int v=int(a.d[i])-int(b.d[i])-borrow;borrow=v<0;a.d[i]=uchar(v+(borrow?10:0));}return a;}
inline Big shift_b(Big a,int n,thread int &err){if(n<0 || n>=ND){if(!iszero(a))err=1;return zero_b();}CALC_NOUNROLL for(int i=ND-n;i<ND;i++)if(a.d[i])err=1;CALC_NOUNROLL for(int i=ND-1;i>=0;i--)a.d[i]=i>=n?a.d[i-n]:0;return a;}
inline Big add_b(Big a,Big b,thread int &err){uint carry=0;CALC_NOUNROLL for(int i=0;i<ND;i++){uint v=uint(a.d[i])+uint(b.d[i])+carry;a.d[i]=uchar(v%10);carry=v/10;}if(carry)err=1;return a;}
inline Big mul_b(Big a,Big b,thread int &err){
 Big r=zero_b();int wa=width(a),wb=width(b);
 CALC_NOUNROLL for(int i=0;i<wa;i++){uint carry=0;CALC_NOUNROLL for(int j=0;j<wb;j++){int k=i+j;uint v=uint(a.d[i])*uint(b.d[j])+carry;
  if(k<ND){v+=uint(r.d[k]);r.d[k]=uchar(v%10);carry=v/10;}else if(v){err=1;carry=0;}
 }int k=i+wb;CALC_NOUNROLL while(carry && k<ND){uint v=uint(r.d[k])+carry;r.d[k]=uchar(v%10);carry=v/10;k++;}if(carry)err=1;}return r;
}
inline Big divmod_b(Big a,Big b,thread Big &rem,thread int &err){
 Big q=zero_b();rem=zero_b();if(iszero(b)){err=3;return q;}
 CALC_NOUNROLL for(int i=width(a)-1;i>=0;i--){rem=shift_b(rem,1,err);rem.d[0]=a.d[i];int digit=0;
  CALC_NOUNROLL while(cmp(rem,b)>=0 && digit<10){rem=sub_b(rem,b);digit++;}if(digit>9){err=7;return q;}q.d[i]=uchar(digit);
 }return q;
}
inline Big mod_b(Big a,Big b,thread int &err){
 Big rem=zero_b();if(iszero(b)){err=3;return rem;}
 CALC_NOUNROLL for(int i=width(a)-1;i>=0;i--){rem=shift_b(rem,1,err);rem.d[0]=a.d[i];int digit=0;
  CALC_NOUNROLL while(cmp(rem,b)>=0 && digit<10){rem=sub_b(rem,b);digit++;}if(digit>9){err=7;return rem;}
 }return rem;
}
inline Rat norm(Rat a,thread int &err){
 if(iszero(a.den)){err=3;return a;}if(iszero(a.n)){a.den=one_b();a.sign=0;return a;}
 Big x=a.n,y=a.den;int steps=0;
 CALC_NOUNROLL while(!iszero(y) && !err && steps++<1000){Big rem=mod_b(x,y,err);x=y;y=rem;}
 if(!iszero(y))err=7;
 if(cmp(x,one_b())!=0 && !err){Big rem;a.n=divmod_b(a.n,x,rem,err);a.den=divmod_b(a.den,x,rem,err);}
 if(width(a.n)>60 || width(a.den)>60)err=1;return a;
}
inline Rat one_d(){Rat a;a.n=one_b();a.den=one_b();a.sign=1;return a;}
inline Rat add_d(Rat a,Rat b,bool subtract,thread int &err){
 if(subtract)b.sign=-b.sign;Rat r;r.den=mul_b(a.den,b.den,err);r.sign=a.sign;
 Big x=mul_b(a.n,b.den,err),y=mul_b(b.n,a.den,err);
 if(a.sign==b.sign)r.n=add_b(x,y,err);else if(cmp(x,y)>=0)r.n=sub_b(x,y);else{r.n=sub_b(y,x);r.sign=b.sign;}
 return r;
}
inline Rat mul_d(Rat a,Rat b,thread int &err){Rat r;r.n=mul_b(a.n,b.n,err);r.den=mul_b(a.den,b.den,err);r.sign=a.sign*b.sign;return r;}
inline Rat pow_d(Rat a,Rat b,thread int &err){
 if(cmp(b.den,one_b())!=0 || width(b.n)>2){err=4;return a;}int e=int(b.n.d[0])+10*int(b.n.d[1]);if(e>64){err=4;return a;}
 Rat r=one_d();int k=e;CALC_NOUNROLL while(k>0 && !err){if(k&1)r=mul_d(r,a,err);k>>=1;if(k)a=mul_d(a,a,err);}
 if(b.sign<0){Big tmp=r.n;r.n=r.den;r.den=tmp;}return norm(r,err);
}
inline void emit(Rat a,device int *result,thread int &err){
 uchar digits[264];int used=0,places=0;Big rem;Big q=divmod_b(a.n,a.den,rem,err);
 CALC_NOUNROLL for(int i=width(q)-1;i>=0;i--)digits[used++]=q.d[i];
 CALC_NOUNROLL while(!iszero(rem) && places<200 && !err){rem=shift_b(rem,1,err);int digit=0;CALC_NOUNROLL while(cmp(rem,a.den)>=0 && digit<10){rem=sub_b(rem,a.den);digit++;}if(digit>9){err=7;break;}digits[used++]=uchar(digit);places++;}
 if(!iszero(rem))err=2;result[0]=err;CALC_NOUNROLL for(int j=1;j<268;j++)result[j]=0;
 if(!err){result[1]=a.sign;result[2]=places;result[3]=used;CALC_NOUNROLL for(int j=0;j<used;j++)result[j+4]=int(digits[used-1-j]);}
}
'''
_SOURCE = r'''
if(thread_position_in_grid.x!=0)return;
int err=left[0];if(right[0])err=right[0];constexpr int op=OP;Rat a,b,r;
a.n=zero_b();a.den=one_b();a.sign=left[1];b.n=zero_b();b.den=one_b();b.sign=right[1];
CALC_NOUNROLL for(int j=0;j<ND;j++){a.n.d[j]=uchar(left[j+2]);a.den.d[j]=uchar(left[j+130]);b.n.d[j]=uchar(right[j+2]);b.den.d[j]=uchar(right[j+130]);}
if(op==0){err=0;r.n=zero_b();r.den=shift_b(one_b(),instruction[1],err);r.sign=1;CALC_NOUNROLL for(int j=0;j<ND;j++)r.n.d[j]=uchar(literal[j]);}
else if(!err){if(op==1 || op==2)r=add_d(a,b,op==2,err);else if(op==3)r=mul_d(a,b,err);else if(op==4)err=7;else if(op==5)r=pow_d(a,b,err);else if(op==6){r=a;r.sign=-r.sign;}else err=7;}
result[0]=err;CALC_NOUNROLL for(int j=1;j<258;j++)result[j]=0;
if(!err){result[1]=r.sign;CALC_NOUNROLL for(int j=0;j<ND;j++){result[j+2]=int(r.n.d[j]);result[j+130]=int(r.den.d[j]);}}
'''
_RENDER = r'''
if(thread_position_in_grid.x!=0)return;
int err=value[0];Rat a;a.n=zero_b();a.den=one_b();a.sign=value[1];CALC_NOUNROLL for(int j=0;j<ND;j++){a.n.d[j]=uchar(value[j+2]);a.den.d[j]=uchar(value[j+130]);}
if(err){result[0]=err;CALC_NOUNROLL for(int j=1;j<268;j++)result[j]=0;}else emit(a,result,err);
'''
_NORMALIZE = r'''
if(thread_position_in_grid.x!=0)return;
int err=value[0];Rat a;a.n=zero_b();a.den=one_b();a.sign=value[1];
CALC_NOUNROLL for(int j=0;j<ND;j++){a.n.d[j]=uchar(value[j+2]);a.den.d[j]=uchar(value[j+130]);}
if(!err)a=norm(a,err);
result[0]=err;CALC_NOUNROLL for(int j=1;j<258;j++)result[j]=0;
if(!err){result[1]=a.sign;CALC_NOUNROLL for(int j=0;j<ND;j++){result[j+2]=int(a.n.d[j]);result[j+130]=int(a.den.d[j]);}}
'''
_GEMM_CARRY = r'''
if(thread_position_in_grid.x!=0)return;
int err=0;
// Validate each literal before multiplying; a zero product cannot hide an invalid operand.
CALC_NOUNROLL for(int p=0;p<2;p++){Rat v;v.n=zero_b();v.den=shift_b(one_b(),scales[p],err);v.sign=1;
 CALC_NOUNROLL for(int j=0;j<ND;j++)v.n.d[j]=uchar(literals[p*ND+j]);norm(v,err);}
Rat a;a.n=zero_b();a.den=shift_b(one_b(),scales[0]+scales[1],err);a.sign=1;uint carry=0;
CALC_NOUNROLL for(int i=0;i<ND;i++){uint v=uint(convolution[i])+carry;a.n.d[i]=uchar(v%10);carry=v/10;}
if(carry)err=1;
result[0]=err;CALC_NOUNROLL for(int j=1;j<258;j++)result[j]=0;
if(!err){result[1]=a.sign;CALC_NOUNROLL for(int j=0;j<ND;j++){result[j+2]=int(a.n.d[j]);result[j+130]=int(a.den.d[j]);}}
'''


@lru_cache(maxsize=4)
def _kernel(kind='arithmetic'):
    import mlx.core as mx
    names={'arithmetic':['left','right','literal','instruction'],'render':['value'],'normalize':['value'],'gemm':['convolution','literals','scales']}[kind]
    return mx.fast.metal_kernel(name='amoeba_exact_rational_'+kind,input_names=names,output_names=['result'],header=_HEADER,
        source={'arithmetic':_SOURCE,'render':_RENDER,'normalize':_NORMALIZE,'gemm':_GEMM_CARRY}[kind],compile_options={'math_mode':'safe'})


def evaluate(expr,backend='gpu'):
    import mlx.core as mx
    try:root,ops,literals,scales=program(expr)
    except ValueError as e:return None,str(e)
    # Explicit GPU stream is essential: a caller's default device may be CPU.
    with mx.stream(mx.gpu):
        use_gemm=backend=='gemm' and ops==[0,0,3]
        expr['arithmetic_execution']='gpu_gemm' if use_gemm else 'gpu_metal'
        if use_gemm:
            # Each dot product has at most 60 products <=81: every FP32 sum is
            # an exact integer <=4860, far below the 2**24 exact-integer limit.
            a=literals[0];b=literals[1]
            toeplitz=mx.array([[a[i-j] if 0<=i-j<128 else 0 for j in range(128)] for i in range(128)],dtype=mx.float32)
            # N=16 makes this matrix-matrix GEMM rather than a GEMV dispatch.
            # Extra columns are deliberate padding for this single-calculation experiment.
            rhs=mx.contiguous(mx.broadcast_to(mx.array(b,dtype=mx.float32)[:,None],(128,16)))
            conv=(toeplitz@rhs)[:,0]
            value=_kernel('gemm')(inputs=[conv,mx.array(literals,dtype=mx.int32),mx.array(scales,dtype=mx.int32)],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(258,)],output_dtypes=[mx.int32])[0]
            value=_kernel('normalize')(inputs=[value],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(258,)],output_dtypes=[mx.int32])[0]
            out=_kernel('render')(inputs=[value],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(268,)],output_dtypes=[mx.int32])[0]
        else:
            # Host walks syntax only; every numerical intermediate stays opaque on GPU.
            stack=[];empty=mx.zeros((258,),dtype=mx.int32);blank=mx.zeros((128,),dtype=mx.int32)
            for i,op in enumerate(ops):
                if op==0:left=right=empty
                elif op==6:left=stack.pop();right=empty
                else:right=stack.pop();left=stack.pop()
                if op==4:
                    # Reciprocal is a device-side field permutation; use the checked
                    # multiplication primitive rather than a separate division body.
                    right=mx.concatenate([right[:2],right[130:258],right[2:130]])
                    op=3
                value=_kernel()(template=[('OP',op)],inputs=[left,right,mx.array(literals[i],dtype=mx.int32) if op==0 else blank,mx.array([op,scales[i]],dtype=mx.int32)],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(258,)],output_dtypes=[mx.int32])[0]
                value=_kernel('normalize')(inputs=[value],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(258,)],output_dtypes=[mx.int32])[0]
                stack.append(value)
            out=_kernel('render')(inputs=[stack[0]],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(268,)],output_dtypes=[mx.int32])[0]
        data=out.tolist()
    if data[0]:return None,{1:'gpu_digit_bound',2:'non_terminating_quotient',3:'division_by_zero',4:'exponent_not_small_integer',7:'gpu_invalid_program'}.get(data[0],'gpu_error')
    body=''.join(str(v) for v in data[4:4+data[3]][::-1]);places=data[2]
    if places:
        body=body.rjust(places+1,'0');body=body[:-places]+'.'+body[-places:]
    return ('-' if data[1]<0 else '')+body,None
