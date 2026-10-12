"""Eight fixed public records; denominator ambiguity, no departure labels.

The integer-minute/second sum checks are hypotheses, not documented properties
of this airline-schedule population. They must not select a production count.
"""
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/external/eurocontrol_performance_20261011/pun_apt_data.json'
EXPECTED = 'c239cd2fc7ae8b274fad56ab100d7bf4401e07bd238e4f7e8d91f4048318e4f1'
OUT = ROOT / 'VALIDATION/reports/punctuality_counts_20261012'
DATES = [f'{year}-{month:02d}-{day:02d}' for year in [2025, 2026]
         for month in [1, 7] for day in [10, 20]]
DESIGN = dict(airport='Rome Fiumicino', dates=DATES,
    denominator_bounds=[1, 3000], observation='Departure Punctuality %',
    precision='half the last published decimal unit; nearest numerator',
    sum_probes='integer minute and integer second sums, unverified granularity',
    selection='fixed dates and airport before source-value inspection',
    no_labels=True, no_model_fit=True, no_cv=True, no_count_selection=True)


def artifact(path):
    return dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def half_unit(value):
    return Decimal(10) ** value.as_tuple().exponent / 2


def compatible(value, error):
    return abs(value - value.to_integral_value(rounding=ROUND_HALF_EVEN)) <= error


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'identity.json').exists(), 'Preserve completed probe'
    identity = dict(design=DESIGN, source=artifact(SOURCE), runner=artifact(Path(__file__)),
                    at=datetime.now(timezone.utc).isoformat())
    assert identity['source']['sha256'] == EXPECTED
    (OUT / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    data = json.loads(SOURCE.read_text())
    header = data[0]
    ix = {name: header.index(name) for name in
          ['Date', 'Airport', 'Departure Punctuality %', 'Avg Departure Schedule Delay']}
    results = []
    for row in data[2:]:
        if row[ix['Airport']] != DESIGN['airport'] or row[ix['Date']] not in DATES:
            continue
        fraction = Decimal(row[ix['Departure Punctuality %']].strip())
        mean = Decimal(row[ix['Avg Departure Schedule Delay']].strip())
        candidates = []
        for n in range(1, 3001):
            numerator = (n * fraction).to_integral_value(rounding=ROUND_HALF_EVEN)
            if abs(numerator / n - fraction) > half_unit(fraction):
                continue
            candidates.append(dict(denominator=n, numerator=int(numerator),
                integer_minute_sum_compatible=compatible(n * mean, n * half_unit(mean)),
                integer_second_sum_compatible=compatible(60 * n * mean, 60 * n * half_unit(mean))))
        assert candidates, 'No count interpretation within the fixed bounds'
        results.append(dict(day=row[ix['Date']], published_fraction=str(fraction),
            published_mean_minutes=str(mean), candidates=candidates,
            candidate_count=len(candidates),
            minute_compatible_count=sum(x['integer_minute_sum_compatible'] for x in candidates),
            second_compatible_count=sum(x['integer_second_sum_compatible'] for x in candidates)))
    assert sorted(x['day'] for x in results) == sorted(DATES)
    output = dict(identity=artifact(OUT / 'identity.json'), records=results,
        conclusion='Compatibility and ambiguity only; no source population or count has been established.',
        caveats=['Decimal precision is interpreted as nearest rounding; formatting/truncation is unverified.',
                 'A reduced fraction and its multiples represent the same percentage.',
                 'Schedule population, daily calendar and time granularity need independent checks.',
                 'Neither the supplied movement count nor a departure label was read.'])
    (OUT / 'result.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps([{k: r[k] for k in ['day', 'candidate_count', 'minute_compatible_count',
                                       'second_compatible_count']} for r in results]))


if __name__ == '__main__':
    main()
