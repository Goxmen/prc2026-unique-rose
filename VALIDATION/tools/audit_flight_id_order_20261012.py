"""Rebuild the flight-ID chronology groups from original input columns."""
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'tmp/research_20261012/signal_scout/flight_id_order'
OUT = ROOT / 'VALIDATION/reports/source_clocks_20261012/flight_id_order_audit.json'


def art(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    assert not OUT.exists()
    frozen = json.loads((RUN/'design.json').read_text())
    result = json.loads((RUN/'result.json').read_text())
    assert result['design_sha256'] == art(RUN/'design.json')['sha256']
    assert result['source_sha256'] == art(ROOT/'VALIDATION/tools/research_flight_id_order_20261012.py')['sha256']
    dates = frozen['design']['query_dates']
    checked = 0
    sources = []
    with duckdb.connect(config={'threads': 1, 'memory_limit': '256MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        for record, report in zip(frozen['source_records'], result['reports'], strict=True):
            path = ROOT/record['path']
            assert report['source_path'] == record['path']
            assert path.stat().st_size == record['bytes'] and path.stat().st_mtime_ns == record['mtime_ns']
            # Read just IDs, endpoint identities and the two original plan clocks.
            con.execute('''CREATE OR REPLACE TEMP TABLE x AS SELECT
                strftime(MVT_TIME_UTC_mvt,'%Y-%m') m,
                strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') d,
                FLIGHT_ID_mvt id, PHASE_mvt phase,ADEP_mvt origin,ADES_mvt destination,
                epoch(IOBT_flt) iobt,epoch(EOBT_1_flt) eobt
                FROM read_parquet(?)''', [str(path)])
            con.execute('''CREATE OR REPLACE TEMP TABLE f AS SELECT m,id,
                count(*) FILTER(WHERE phase='DEP') nd,count(*) FILTER(WHERE phase='ARR') na,
                count(*) FILTER(WHERE phase IS NULL OR phase NOT IN ('DEP','ARR')) other,
                count(DISTINCT struct_pack(origin:=origin,destination:=destination)) routes,
                count(DISTINCT iobt) ni,count(DISTINCT eobt) ne,
                min(origin) origin,min(d) FILTER(WHERE phase='DEP') d,
                min(iobt) FILTER(WHERE phase='DEP') iobt,min(eobt) FILTER(WHERE phase='DEP') eobt
                FROM x WHERE isfinite(id) AND id>0 AND id<=9007199254740992 AND id=floor(id)
                GROUP BY m,id''')
            frame = con.execute('''SELECT m,id,origin,d,iobt,eobt FROM f
                WHERE nd=1 AND na<=1 AND other=0 AND routes=1 AND ni<=1 AND ne<=1''').df()
            assert set(frame.m) == {r['month'] for r in report['monthly']}
            for month in report['monthly']:
                part = frame.loc[frame.m.eq(month['month'])]
                assert len(part) == month['eligible_departures']
                for clock in ['iobt', 'eobt']:
                    known = part.dropna(subset=[clock])
                    np.testing.assert_allclose(spearmanr(known.id, known[clock]).statistic,
                                               month['clocks'][clock]['rho'], atol=2e-12, rtol=0)
            for group in report['chronology_groups']:
                assert group['day'] in dates
                part = frame.loc[frame.m.eq(group['month']) & frame.d.eq(group['day'])]
                if group['airport'] != 'ALL':
                    part = part.loc[part.origin.eq(group['airport'])]
                for clock in ['iobt', 'eobt']:
                    known = part.dropna(subset=[clock]); expected = group['clocks'][clock]
                    assert len(known) == expected['n']
                    if expected['rho'] is None:
                        assert len(known)<100 or known.id.nunique()<2 or known[clock].nunique()<2
                    else:
                        np.testing.assert_allclose(spearmanr(known.id, known[clock]).statistic,
                                                   expected['rho'], atol=2e-12, rtol=0)
                    checked += 1
            sources.append(art(path))
    receipt = dict(status='pass', source=art(__file__), design=art(RUN/'design.json'), result=art(RUN/'result.json'),
                   source_files=sources, checked_group_clock_pairs=checked,
                   all_eligible_counts_and_monthly_and_fixed_date_correlations_rebuilt=True,
                   calculation='Independent narrow-column SQL eligibility and SciPy Spearman ranks',
                   scope='Chronology arithmetic and eligible counts; original endpoint/format descriptive tables retain their reviewed pilot implementation.',
                   interpretation='Almost exact relation to supplied IOBT may be redundant. No creation-time interpretation, new predictive information, or validation gain established.',
                   no_target_column_projected=True, no_fit=True, no_cv=True)
    OUT.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(status='pass', group_clock_pairs=checked)))


if __name__ == '__main__':
    main()
