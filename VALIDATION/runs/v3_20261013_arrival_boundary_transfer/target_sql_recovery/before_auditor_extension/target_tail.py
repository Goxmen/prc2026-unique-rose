# AST-derived original prepare target tail; exactly one SQL AS alias repair.
def recovered_target_tail(ident, meta, packets, started, deadline):
    c = connection()
    c.execute("SET TimeZone='UTC'")
    rawpaths = [str(verify(a)) for a in ident['raw']]
    truth = c.execute("SELECT MVT_ID_mvt::BIGINT mvt_id,PHASE_mvt phase,\n        epoch_ms(MVT_TIME_UTC_mvt)::BIGINT event_ms,TAXITIME_SEC_mvt::DOUBLE taxi_sec,\n        CASE WHEN PHASE_mvt='DEP' THEN ADEP_mvt ELSE ADES_mvt END airport,\n        strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') AS day\n        FROM read_parquet(?) WHERE PHASE_mvt IN ('DEP','ARR')\n        AND strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') IN (SELECT unnest(?))", [rawpaths, DAYS]).df()
    assert truth.mvt_id.is_unique and set(truth.day) == set(DAYS)
    joined = meta.merge(truth, on=['mvt_id', 'phase', 'day', 'airport', 'event_ms'], how='left', validate='one_to_one', indicator=True)
    assert joined._merge.eq('both').all()
    cached = pd.read_parquet(verify(ident['arrivals'][0]), columns=['mvt_id', 'inblock_ms'])
    arr_truth = truth[truth.phase.eq('ARR')].merge(cached, on='mvt_id', validate='one_to_one', how='left')
    expected = np.isfinite(arr_truth.taxi_sec) & arr_truth.taxi_sec.ge(0)
    assert np.array_equal(arr_truth.inblock_ms.notna().to_numpy(), expected.to_numpy())
    error = np.abs(arr_truth.loc[expected, 'inblock_ms'] - (arr_truth.loc[expected, 'event_ms'] + 1000 * arr_truth.loc[expected, 'taxi_sec']))
    assert len(error) == 0 or error.max() <= 1.0
    by_id = cached.set_index('mvt_id').inblock_ms
    joined['block_ms'] = joined.event_ms - 1000 * joined.taxi_sec
    isarr = joined.phase.eq('ARR')
    joined.loc[isarr, 'block_ms'] = joined.loc[isarr, 'mvt_id'].map(by_id)
    joined['label_available'] = np.isfinite(joined.block_ms)
    labelpath = OUT / 'labels.parquet'
    joined.drop(columns='_merge').to_parquet(labelpath, index=False)
    labeled = joined.set_index('mvt_id')
    supervised = []
    for packet in packets:
        assert time.monotonic() <= deadline, 'Fixed five-minute preparation budget exceeded'
        frame = pd.read_parquet(verify(packet['flights']))
        times = np.load(verify(packet['times']), allow_pickle=False, mmap_mode='r')
        target = np.full(len(times), np.nan)
        for row in frame.itertuples():
            label = labeled.loc[row.mvt_id]
            if label.label_available:
                sl = slice(int(row.start), int(row.start + row.count))
                target[sl] = boundary_targets(times[sl], label.block_ms, row.phase)
        path = (ROOT / packet['features']['path']).parent / 'targets.npy'
        np.save(path, target, allow_pickle=False)
        supervised.append(dict(**packet, targets=art(path), finite_targets=int(np.isfinite(target).sum()), targets_below_001=int(np.sum(target < 0.01)), targets_above_099=int(np.sum(target > 0.99))))
    counts, passed = ([], True)
    for half in HALVES:
        airports = []
        for phase in ['DEP', 'ARR']:
            selected = joined[joined.half.eq(half) & joined.phase.eq(phase)]
            valid = selected[selected.train_selected & selected.label_available]
            counts.append(dict(half=half, phase=phase, eligible=len(selected), sampled=int(selected.train_selected.sum()), finite_supervised_sample=len(valid), finite_evaluation=int(selected.label_available.sum())))
            passed &= len(valid) >= 500
            airports.append(set(valid.airport))
        passed &= len(airports[0] & airports[1]) >= 2
    c.close()
    write(OUT / 'prepared.json', dict(identity=art(OUT / 'identity.json'), inputs=art(OUT / 'inputs.json'), labels=art(labelpath), packets=supervised, counts=counts, coverage_gate=bool(passed), arrival_raw_negative=int(arr_truth.taxi_sec.lt(0).sum()), arrival_raw_nonfinite=int((~np.isfinite(arr_truth.taxi_sec)).sum()), arrival_cache_null=int(arr_truth.inblock_ms.isna().sum()), peak_rss_MB=peak_mb(), elapsed_seconds=time.monotonic() - started))
    if not passed:
        write(OUT / 'closed.json', dict(state='inconclusive_insufficient_coverage', prepared=art(OUT / 'prepared.json')))
    print(json.dumps({'prepared': art(OUT / 'prepared.json'), 'coverage_gate': bool(passed)}), flush=True)
