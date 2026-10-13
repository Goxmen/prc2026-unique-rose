"""Single diagnosed input-only support and synthetic checks; no posterior run."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import portable_public_tilt_recovery_20261013 as producer
from tools import audit_portable_public_tilt_recovery_20261013 as auditor
OUT=ROOT/'VALIDATION/reports/portable_tilt_recovery_preflight_20261013.json'


def main():
    assert not OUT.exists()
    original=[]
    for target in [0.,.25,1.,1.75,2.]:
        values=np.array([0.,1.,2.]);mass=np.array([.2,.3,.5])
        expected=producer.ORIGINAL_TILT(values,mass,target);events=[]
        actual=producer.recovered_tilt(values,mass,target,events.append)
        checked=auditor.audit_tilt(values,mass,target)
        assert expected.tobytes()==actual.tobytes()==checked.tobytes() and not events
        original.append(target)
    record=auditor.read(producer.DIAGNOSIS);row=record['row']
    params=auditor.read(auditor.verify(record['parameters']))['parameters'][row['airport']]
    values,mass=auditor.atoms(row,params)
    np.testing.assert_array_equal(values,record['support']['values']);np.testing.assert_array_equal(mass,record['support']['base_weights'])
    events=[];producer.recovered_tilt(values,mass,row['prior'],events.append);assert len(events)==1
    event=dict(row=row,**events[0]);check=auditor.event_check(event,row,params)
    failures=[]
    def reject(name,value):
        try:auditor.event_check(value,row,params)
        except AssertionError:failures.append(name)
        else:raise AssertionError(name)
    reject('wrong target',dict(event,target=row['prior']+1))
    reject('wrong mean tolerance',dict(event,unchanged_atol=1.))
    reject('wrong iteration count',dict(event,iterations=54))
    wrong=event['weights'].copy();wrong[0]+=.001;reject('wrong weights',dict(event,weights=wrong))
    wrong=event['base_weights'].copy();wrong[0]*=2;reject('wrong prior masses',dict(event,base_weights=wrong))
    for values,mass,target in [(np.array([0.,1.,2.]),np.array([1.,0.,0.]),1.99),
                               (np.array([0.,1.,2.]),np.array([0.,0.,1.]),.01)]:
        events=[];fixed=producer.recovered_tilt(values,mass,target,events.append)
        independent,root=auditor.brent_weights(values,mass,target)
        assert len(events)==1;np.testing.assert_allclose(fixed,independent,atol=1e-12,rtol=0)
        assert auditor.mean_pass(values,fixed,target)
    report=dict(status='pass',source=auditor.art(Path(__file__)),producer=auditor.art(Path(producer.__file__)),
        auditor=auditor.art(Path(auditor.__file__)),diagnosis=auditor.art(producer.DIAGNOSIS),
        original_success_cases_byte_identical=original,diagnosed_event=check,synthetic_both_sided_extensions=True,
        corruptions_rejected=failures,no_posterior_inference=True,no_labels=True,no_fit=True)
    OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='pass',receipt=auditor.art(OUT),diagnosed_weight_error=check['max_weight_error'])))


if __name__=='__main__':main()
