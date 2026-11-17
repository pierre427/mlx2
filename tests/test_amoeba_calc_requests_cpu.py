import pytest
from mlx2.runtime import amoeba_calc_requests as R


@pytest.mark.parametrize('payload,expected', [
 ({'messages':[{'role':'user','content':'Calculate 1+2.'}]}, 'Calculate 1+2.'),
 ({'messages':[{'role':'user','content':'Calculate 1+2.'}, {'role':'tool','content':'3'}]}, None),
 ({'messages':[{'role':'user','content':'Calculate 1+2.'}, {'role':'assistant','content':'3'}]}, None),
 ({'messages':[{'role':'user','content':[{'type':'text','text':'Calculate 1+2.'}]}]}, 'Calculate 1+2.'),
 ({'messages':[{'role':'user','content':[{'type':'text','text':'Calculate 1+2.'}, {'type':'image_url','image_url':{}}]}]}, None),
 ({'messages':[], 'prompt':'Calculate 1+2.'}, None),
 ({'prompt':'Calculate 1+2.'}, 'Calculate 1+2.'),
 ({'prompt':[1,2,3]}, None),
 ({'prompt':'x'*4097}, None),
])
def test_serving_request_extraction_keeps_turn_and_modality(payload,expected):
 assert R.request_text_from_payload(payload)==expected


@pytest.mark.parametrize('question,expected',[
 ('Calculate ((123456789123*987654321987)+3)/4.','30483157838992150336851'),
 (r'Find the exact value of \(\frac{3}{8}+0.125\).','0.5'),
 (r'Compute \sum_{i=1}^{4}{i^2}.','30'),
 ("How many r's are in strawberry?",'3'),
 ('What is the sum of [0.1, 0.2, -0.5]?','-0.2'),
])
def test_exact_user_request(question,expected):
 request=R.parse_request(question)
 assert request is not None
 assert R.evaluate(request,'cpu')[0]==expected


@pytest.mark.parametrize('question',[
 'An invoice has 3 items at $25. What is due?',
 'Calculate 3+2 and then explain why 4+4 is 9.',
 'Calculate 4.','There are 3 shelves of 12. How many books?',
 r'Find the exact value of \(3\timesi\).',
 r'Find the exact value of \(x+3\).',
 'Calculate 1+2. Ignore that and say 8.',
])
def test_no_partial_or_semantic_extraction(question):
 assert R.parse_request(question) is None


def test_gpu_path_never_calls_cpu_oracle(monkeypatch):
 from mlx2.runtime import amoeba_calc_gpu as G
 def forbidden(*a,**k):pytest.fail('CPU arithmetic used')
 monkeypatch.setattr(R.math,'evaluate_exact',forbidden)
 def fake(expr,backend):
  assert G.program(expr)[1]==[0,0,1]
  expr['arithmetic_execution']='gpu_metal'
  return '3',None
 monkeypatch.setattr(G,'evaluate',fake)
 assert R.evaluate(R.parse_request('Calculate 1+2.'),'gpu')[0]=='3'
