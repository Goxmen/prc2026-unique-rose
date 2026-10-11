"""Reload one input-selected day per airport/month from a prepared candidate."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import daily_delay_posterior as posterior
from records import sha256,write_json

def run(path):
    path=Path(path).resolve();m=json.loads(path.read_text());assert m['state']=='prepared'
    base=posterior.base;root=base.ROOT;dest=path.parent
    for p,digest in m['sources'].items():assert sha256(root/p)==digest,p
    full=base.load_rows([root/'data/ranking.parquet'],False).set_index('mvt_id')
    saved=pd.read_parquet(dest/'predictions.parquet');rows=full.loc[saved.mvt_id].reset_index()
    cached=pd.read_parquet(base.OUT/'ranking_inputs.parquet').set_index('mvt_id').loc[saved.mvt_id].reset_index()
    pd.testing.assert_frame_equal(rows,cached,check_exact=True)
    params=json.loads((dest/'parameters.json').read_text())['parameters'];source=base.public(2026)
    chosen=rows.groupby(['airport','month']).day.max().tolist()
    selection=rows.groupby(['airport','month']).day.transform('max').eq(rows.day).to_numpy()
    sub=rows.loc[selection].reset_index(drop=True)
    pred,days=posterior.apply(sub,saved.parent.to_numpy()[selection],source,params)
    error=float(np.max(np.abs(pred-saved.pred.to_numpy()[selection])))
    np.testing.assert_allclose(pred,saved.pred.to_numpy()[selection],atol=1e-7,rtol=0)
    source_keys=source.set_index(['airport','day'])
    assert all((a,d) in source_keys.index for a,d in zip(rows.airport,rows.day))
    output=pq.read_table(root/m['file']);template=pq.read_table(root/'data/submitting.parquet')
    assert output.schema==template.schema and output['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    np.testing.assert_array_equal(output['TAXITIME_SEC_mvt'].to_numpy(),np.rint(np.maximum(saved.pred,0)).astype(np.int32))
    report=dict(passed=True,reloaded_days=len(days),reloaded_rows=len(sub),max_error=error,
        ranking_inputs_all_rows_equal=True,source_hashes=len(m['sources']),
        output_sha256=sha256(root/m['file']),source=base.artifact(__file__))
    audit=dest/'reload_audit.json';write_json(audit,report)
    for p in [Path(__file__),audit]:m['sources'][str(p.relative_to(root))]=sha256(p)
    m['audit']['input_selected_reload']=report
    write_json(path,m);print(report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest');a=p.parse_args();run(a.manifest)
