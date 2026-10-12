"""Independent exact-rational check of the fixed eight-record JSON probe.

Use rational interval endpoints and integer floor/ceil, not Decimal rounding
or the original compatibility functions. No movement rows or labels are read.
"""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/reports/punctuality_counts_20261012'

def art(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def ceil(v):return -((-v.numerator)//v.denominator)
def floor(v):return v.numerator//v.denominator

def interval(token):
    token=token.strip()
    assert 'e' not in token.lower(),'Fixed source tokens use ordinary decimal notation'
    places=len(token.split('.')[1]) if '.' in token else 0
    value=Fraction(token);half=Fraction(1,2*10**places)
    return value-half,value+half

def has_integer(lo,hi):return ceil(lo)<=floor(hi)

def main():
    target=OUT/'independent_audit.json';assert not target.exists()
    identity=json.loads((OUT/'identity.json').read_text())
    result=json.loads((OUT/'result.json').read_text())
    assert result['identity']==art(OUT/'identity.json')
    source=ROOT/identity['source']['path'];assert art(source)==identity['source']
    assert art(ROOT/identity['runner']['path'])==identity['runner']
    data=json.loads(source.read_text());header=data[0];ix={k:i for i,k in enumerate(header)}
    records={(r[ix['Date']],r[ix['Airport']]):r for r in data[2:]}
    assert len(result['records'])==8 and {r['day'] for r in result['records']}==set(identity['design']['dates'])
    summaries=[]
    for recorded in result['records']:
        row=records[(recorded['day'],identity['design']['airport'])]
        p=row[ix['Departure Punctuality %']].strip();m=row[ix['Avg Departure Schedule Delay']].strip()
        assert p==recorded['published_fraction'] and m==recorded['published_mean_minutes']
        pl,ph=interval(p);ml,mh=interval(m);expected=[]
        for n in range(1,3001):
            low,high=ceil(pl*n),floor(ph*n)
            if low>high:continue
            assert low==high,'Probe records one numerator per denominator; detect a wider compatible interval'
            expected.append(dict(denominator=n,numerator=low,
                integer_minute_sum_compatible=has_integer(ml*n,mh*n),
                integer_second_sum_compatible=has_integer(ml*60*n,mh*60*n)))
        assert expected==recorded['candidates']
        counts=dict(candidate_count=len(expected),
            minute_compatible_count=sum(c['integer_minute_sum_compatible'] for c in expected),
            second_compatible_count=sum(c['integer_second_sum_compatible'] for c in expected))
        assert all(recorded[k]==v for k,v in counts.items())
        summaries.append(dict(day=recorded['day'],**counts))
    value=dict(status='pass_exact_rational_arithmetic',source=art(Path(__file__)),
        inputs=[identity['source'],art(OUT/'identity.json'),art(OUT/'result.json'),identity['runner']],
        records=summaries,denominators_checked=24000,
        scope='Independent rational interval/integer intersection reconstruction under the original nearest-rounding and reported-precision assumptions; no claim those assumptions or source-population equivalence are established.',
        no_count_selection=True,no_movement_or_departure_label_read=True,no_cv=True)
    target.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(summaries))

if __name__=='__main__':main()
