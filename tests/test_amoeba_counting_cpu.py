"""Exact literal counting; ambiguous requests refuse rather than invent targets."""
import pytest
from mlx2.runtime import amoeba_counting as C

@pytest.mark.parametrize('question,answer',[
 ("How many r's are in strawberry?",'3'),
 ("How many r's are there in the word strawberry?",'3'),
 ("Count 's' in 'Mississippi'.",'4'),
 ("How many times does the letter 'r' occur in the word 'strawberry'?",'3'),
 ("Count 'r' in 'stRawbeRry' case-sensitive.",'1'),
 ("Count 'r' in 'stRawbeRry' ignore case.",'3'),
 ('''Count " " in "a b c".''','2'),
 ('How many entries are in [12, -5, 12, 0.25]?','4'),
 ('Count items in []','0'),
 ('What is the sum of [0.1, 0.2, -3.5, 10]?','6.8'),
 ('Sum [123456789123.123456789, 0.000000001]','123456789123.12345679'),
 ('Sum []','0'),
])
def test_literal_requests(question,answer):
 spec=C.parse_request(question)
 assert spec is not None
 assert C.evaluate(spec,'cpu')[0]==answer

@pytest.mark.parametrize('question',[
 "How many r's are in the words you used?", "Count 'rr' in strawberry", 'Sum [1, 2, three]',
 'Sum [1,,2]', 'Sum [2**10, 4]', 'Count items in [1,2]; import os', 'How many apples were sold?',
])
def test_ambiguous_or_executable_requests_refuse(question):
 assert C.parse_request(question) is None
