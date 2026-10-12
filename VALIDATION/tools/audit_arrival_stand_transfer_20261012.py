"""Independent SQL and scalar fold-isolation audit of the stand-transfer pilot."""
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'VALIDATION/runs/v3_20261012_arrival_stand_transfer'
MONTHS=['2025-03','2025-09','2025-10','2025-11']


def art(path):
    path=Path(path)
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    output=RUN/'audit.json';assert not output.exists()
    read=lambda path:json.loads(path.read_text())
    identity,result=read(RUN/'identity.json'),read(RUN/'result.json')
    assert result['identity']==art(RUN/'identity.json') and identity['design']['months']==MONTHS
    for item in [*identity['sources'],*result['artifacts']]:assert art(ROOT/item['path'])==item
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        con.read_parquet(str(ROOT/'data/training_2025-*.parquet')).create_view('source')
        con.read_parquet(str(RUN/'cells.parquet')).create_view('saved')
        con.execute('''CREATE TEMP TABLE allowed AS
            SELECT PHASE_mvt phase,CASE WHEN PHASE_mvt='DEP' THEN ADEP_mvt ELSE ADES_mvt END airport,
            STAND_mvt stand,strftime(MVT_TIME_UTC_mvt,'%Y-%m') AS month,TAXITIME_SEC_mvt y
            FROM source WHERE strftime(MVT_TIME_UTC_mvt,'%Y-%m') IN (SELECT unnest(?))
            AND STAND_mvt IS NOT NULL AND ((PHASE_mvt='ARR' AND TAXITIME_SEC_mvt>=1 AND TAXITIME_SEC_mvt<=7200)
              OR (PHASE_mvt='DEP' AND TAXITIME_SEC_mvt>=1 AND TAXITIME_SEC_mvt<=10800))''',[MONTHS])
        assert con.execute('SELECT count(*) FROM allowed').fetchone()[0]==result['sources_rows']
        con.execute('''CREATE TEMP TABLE reconstructed AS
            SELECT DISTINCT phase,airport,stand,month,
                count(*) OVER (PARTITION BY phase,airport,stand,month) AS n,
                median(y) OVER (PARTITION BY phase,airport,stand,month) AS stand_median,
                count(*) OVER (PARTITION BY phase,airport,month) AS airport_n,
                median(y) OVER (PARTITION BY phase,airport,month) AS airport_median FROM allowed''')
        n,bad=con.execute('''SELECT count(*),sum(CASE WHEN s.n IS DISTINCT FROM r.n
            OR s.stand_median IS DISTINCT FROM r.stand_median OR s.airport_n IS DISTINCT FROM r.airport_n
            OR s.airport_median IS DISTINCT FROM r.airport_median
            OR s.centered IS DISTINCT FROM r.stand_median-r.airport_median THEN 1 ELSE 0 END)
            FROM saved s FULL JOIN (SELECT * FROM reconstructed WHERE n>=20) r
            USING(phase,airport,stand,month)''').fetchone()
        assert n==result['eligible_phase_cells'] and bad==0
    saved=pd.read_parquet(RUN/'cells.parquet')
    lookup={(r.phase,r.airport,r.stand,r.month):(r.centered,r.n) for r in saved.itertuples()}
    observed=pd.read_parquet(RUN/'transfer.parquet')
    keys=sorted({(a,s) for p,a,s,m in lookup if all((phase,a,s,month) in lookup for phase in ['ARR','DEP'] for month in MONTHS)})
    assert len(keys)==result['common_stands'] and len(observed)==4*len(keys)
    differences=[]
    for rec in result['records']:
        held=rec['month'];training=[m for m in MONTHS if m!=held]
        numerator=denominator=0.
        for a,s in keys:
            for month in training:
                others=[m for m in training if m!=month]
                dx=lookup['ARR',a,s,month][0]-sum(lookup['ARR',a,s,m][0] for m in others)/2
                dy=lookup['DEP',a,s,month][0]-sum(lookup['DEP',a,s,m][0] for m in others)/2
                weight=min(lookup['DEP',a,s,month][1],100)
                numerator+=weight*dx*dy;denominator+=weight*dx*dx
        slope=numerator/denominator if denominator else 0.
        assert abs(slope-rec['slope'])<1e-12
        baseline=candidate=weights=0.
        frame=observed.loc[observed.month.eq(held)].set_index(['airport','stand'])
        assert set(frame.index)==set(keys)
        for a,s in keys:
            dx=lookup['ARR',a,s,held][0]-sum(lookup['ARR',a,s,m][0] for m in training)/3
            dy=lookup['DEP',a,s,held][0]-sum(lookup['DEP',a,s,m][0] for m in training)/3
            weight=min(lookup['DEP',a,s,held][1],100)
            row=frame.loc[a,s]
            np.testing.assert_allclose([row.delta_ARR,row.delta_DEP,row.weight,row.prediction],
                [dx,dy,weight,slope*dx],atol=1e-10,rtol=0)
            baseline+=weight*dy*dy;candidate+=weight*(dy-slope*dx)**2;weights+=weight
        np.testing.assert_allclose([baseline/weights,candidate/weights],
            [rec['baseline_mse'],rec['candidate_mse']],atol=1e-8,rtol=0)
        differences.append(dict(month=held,slope=slope,mse_delta=(candidate-baseline)/weights))
    passed=all(r['slope']>0 and r['mse_delta']<0 for r in differences)
    assert passed==result['passes_fixed_transfer_gate']
    value=dict(status='pass',source=art(__file__),identity=art(RUN/'identity.json'),result=art(RUN/'result.json'),
        source_rows=result['sources_rows'],phase_cells=n,common_stands=len(keys),folds=differences,
        independent_window_medians_counts_and_centering=True,scalar_training_only_references_and_slopes=True,
        all_saved_deltas_predictions_and_scores=True,passes_fixed_transfer_gate=passed,
        scope='Training-only feasibility arithmetic. No claim of taxi-model improvement or independent heldout validation.')
    output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');print(json.dumps(value,indent=2))


if __name__=='__main__':main()
