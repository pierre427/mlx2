#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
"""Render the calculator tile's one-page, high-school-level explanation."""
import argparse
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color, white
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/pdf';OUT.mkdir(parents=True,exist_ok=True)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--resident',action='store_true',help='Illustrate the GPU-resident tile')
args=parser.parse_args()
path=OUT/('calculator-tile-explained-resident.pdf' if args.resident else 'calculator-tile-explained.pdf')
c=canvas.Canvas(str(path),pagesize=(612,792))
c.setTitle('The calculator tile - exact math inside an AI response')
c.setAuthor('Pierre Lamy')
c.setSubject('A one-page illustrated explanation of deterministic calculator tiles')
navy=HexColor('#122A43');teal=HexColor('#007F82');orange=HexColor('#E88935');ink=HexColor('#213547');muted=HexColor('#536879');pale=HexColor('#EAF5F3');blue=HexColor('#EBF0F7')

def text(x,y,s,size=10,font='Helvetica',color=ink):
 c.setFillColor(color);c.setFont(font,size);c.drawString(x,y,s)

def para(x,top,w,s,size=10,leading=14,color=ink,bold=False):
 style=ParagraphStyle('p',fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,leading=leading,textColor=color)
 p=Paragraph(s,style);_,h=p.wrap(w,700);p.drawOn(c,x,top-h);return h

def box(x,y,w,h,fill,r=10,stroke=None):
 c.setFillColor(fill);c.setStrokeColor(stroke or fill);c.roundRect(x,y,w,h,r,stroke=bool(stroke),fill=1)

def arrow(x,y,length=12,color=teal):
 c.setStrokeColor(color);c.setFillColor(color);c.setLineWidth(1.8);c.line(x,y,x+length-4,y)
 p=c.beginPath();p.moveTo(x+length,y);p.lineTo(x+length-5,y+3);p.lineTo(x+length-5,y-3);p.close();c.drawPath(p,fill=1,stroke=0)

# Header
c.setFillColor(navy);c.rect(0,626,612,166,stroke=0,fill=1)
text(36,757,'MLX2  /  CALCULATOR TILE',10,'Helvetica-Bold',HexColor('#7CE0CF'))
text(36,719,'An AI with a built-in calculator',25,'Helvetica-Bold',white)
para(36,697,495,'The language model handles the words. A small add-on handles supported calculations using fixed rules.',12,17,white)
# calculator icon
box(541,746,35,30,teal,5)
for yy in [754,763]:
 for xx in [549,558,567]:box(xx,yy,4,4,white,1)

para(36,606,540,
     'A <b>deterministic (DET) tile</b> is a plug-in. A supported calculation always gives the same result. The tile can work with different language models using the same math rules.'
     if args.resident else
     'A <b>deterministic (DET) tile</b> is a small plug-in: the same supported input gives the same calculated result. It can help without retraining the whole language model.',11,15)

# Worked example
box(36,450,540,93,pale)
text(51,522,'A WORKED EXAMPLE',9,'Helvetica-Bold',teal)
text(51,500,'You ask: Calculate (18.75 × 24) ÷ 3.',13,'Helvetica-Bold')
text(51,473,'18.75 × 24',18,'Helvetica-Bold',navy)
arrow(177,479,26)
text(218,473,'450 ÷ 3',18,'Helvetica-Bold',navy)
arrow(315,479,26)
box(355,461,75,29,teal,7);text(373,470,'150',18,'Helvetica-Bold',white)
text(447,474,'exact answer',10,'Helvetica',muted)

text(36,427,'HOW THE ANSWER GETS INTO THE RESPONSE',10,'Helvetica-Bold',navy)
steps=[
 ('1','Notice','GPU + rules','A gate spots a calculation during writing, or a strict parser recognizes a complete math request.'),
 ('2','Read','CPU','Text becomes a short list of numbers and operations. The parser checks that the request is supported.'),
 ('3','Calculate','GPU','Metal kernels work with integer digits and fractions. This avoids rounding errors within the tile\'s limits.'),
 ('4','Write','CPU + model','The CPU formats the result. The tile makes the model emit its exact answer tokens, then finish or continue.')]
if args.resident:
 steps=[
  ('1','Notice','GPU','The tile looks for a complete math request or a calculation the model has just written.'),
  ('2','Check','GPU','The parser reads every number and operation. An unsupported expression keeps the gate closed.'),
  ('3','Calculate','GPU','Metal kernels use integer digits and fractions. The gate opens only when the tile has an exact, supported result.'),
  ('4','Write','GPU','The tile turns the answer into the model’s tokens. The model emits them and then finishes or continues.')]
for i,(number,title,badge,body) in enumerate(steps):
 x=36+i*139
 box(x,258,123,151,blue if i!=2 else pale)
 c.setFillColor(teal if i==2 else navy);c.circle(x+18,390,10,stroke=0,fill=1)
 text(x+15,387,number,10,'Helvetica-Bold',white)
 text(x+34,386,title,12,'Helvetica-Bold')
 text(x+11,367,badge,8,'Helvetica-Bold',teal)
 para(x+11,355,101,body,9.2,12)
 if i<3:arrow(x+126,333,10)

# Counting graphic
box(36,193,540,49,navy)
text(49,223,'COUNTING WORKS TOO',8,'Helvetica-Bold',HexColor('#7CE0CF'))
text(49,207,'How many r\'s are in',11,'Helvetica',white)
x=176
for ch in 'strawberry':
 color=HexColor('#FFC17E') if ch=='r' else white
 text(x,206,ch,16,'Courier-Bold',color)
 if ch=='r':c.setStrokeColor(color);c.setLineWidth(1);c.line(x,202,x+8,202)
 x+=10
arrow(282,213,22,HexColor('#7CE0CF'))
text(318,206,'3',21,'Helvetica-Bold',HexColor('#FFC17E'))
para(355,226,204,'The GPU compares letters and adds the matches.',9.5,12,white)

text(36,174,'What it can do',11,'Helvetica-Bold',teal)
if args.resident:
 para(36,162,253,'It handles arithmetic, powers, fractions and supported LaTeX. It can also count letters and list items or add a list of numbers.',10,13)
 text(321,174,'Where it needs the model',11,'Helvetica-Bold',teal)
 para(321,162,255,'For word problems, the model still has to choose the right calculation. The tile can help once it recognizes a supported expression.',10,13)
 para(36,95,540,'The <b>resident tile</b> handles parsing, arithmetic and answer tokens on the GPU. The CPU initializes the vocabulary, schedules GPU work and handles requests and responses.',9.6,13,muted)
else:
 para(36,162,253,'Add, subtract, multiply and divide; powers, fractions and supported LaTeX maths; bounded sums and products; letter counts, list counts and list sums.',10,13)
 text(321,174,'What it does not solve',11,'Helvetica-Bold',teal)
 para(321,162,255,'It does not understand every word problem. The model still has to choose the right calculation. Unsupported maths is declined; this is not a general algebra solver.',10,13)
 para(36,95,540,'<b>Where it runs:</b> arithmetic is on the GPU; parsing, formatting and tokenization still use the CPU. It is not fully GPU-only, and speed gains are not guaranteed.',9.6,13,muted)

c.setStrokeColor(HexColor('#D5DFE6'));c.setLineWidth(.7);c.line(36,55,576,55)
edition='Experimental resident tile' if args.resident else 'Experimental implementation'
text(36,41,f'Calculator tile by Pierre Lamy  |  {edition}  |  17 November 2026',8,'Helvetica',muted)
text(36,28,'Free noncommercial use with credit. Commercial use or distributed derivatives require permission.',7.6,'Helvetica',muted)
c.linkURL('mailto:pierre@userid.org',(36,22,576,38),relative=0,thickness=0)
c.showPage();c.save();print(path)
