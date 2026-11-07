/* Pages display arithmetic only. Never used by the scorer or event detector. */
(function (root, factory) {
    const api = factory();
    if (typeof module === 'object' && module.exports) module.exports = api;
    else root.GeoSpecHistoryRate = api;
})(typeof globalThis === 'object' ? globalThis : this, function () {
    'use strict';
    const VERSION = 'pages-recorded-change-v1';
    const COLUMNS = ['date', 'previous_date', 'region', 'method', 'basis', 'current', 'previous', 'elapsed_days',
        'change_per_day', 'status', 'reason', 'station', 'recorded_unit', 'recorded_method_version',
        'current_source_sha256', 'previous_source_sha256', 'clock', 'definition'];
    const finite = v => typeof v === 'number' && Number.isFinite(v) ? v : null;
    const known = v => typeof v === 'string' && v.trim() !== '' && !/^(unknown|unspecified|not[_ ]recorded)$/i.test(v);
    const canonical = v => Array.isArray(v) ? v.map(canonical) : v && typeof v === 'object'
        ? Object.fromEntries(Object.keys(v).sort().map(k => [k, canonical(v[k])])) : v ?? null;
    function dayTime(day) {
        if (typeof day !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(day)) return null;
        const t = Date.parse(day + 'T00:00:00Z');
        return Number.isFinite(t) && new Date(t).toISOString().slice(0, 10) === day ? t : null;
    }
    function endpoint(day, method, basis) {
        const m = day.methods?.[method] || {}, cal = m.calibration || {};
        const value = basis === 'score' ? m.score : m.raw;
        const source = day.source || {};
        let reason = null;
        if (dayTime(day.date) === null) reason = 'INVALID_REPORT_DAY';
        else if (!['LEGACY_LAST_COMMITTED_BLOB', 'REVISION_CURRENT'].includes(source.kind) || !/^[a-f0-9]{64}$/.test(source.sha256 || '')) reason = 'REPORT_PROVENANCE_UNKNOWN';
        else if (day.revision_history_complete === false || (day.revision_count != null && day.revision_count !== 1) || (day.superseded || []).length) reason = 'REVISED_DAY_NOT_COMPARABLE';
        else if (m.recorded?.frozen === true || (m.warnings || []).includes('FROZEN')) reason = 'FROZEN';
        else if (m.raw?.state !== 'MEASURED' || finite(m.raw.value) === null) reason = 'MEASUREMENT_UNKNOWN';
        else if (!value || value.state !== (basis === 'score' ? 'RECORDED' : 'MEASURED') || finite(value.value) === null) reason = 'VALUE_UNKNOWN';
        else if (basis === 'score' && m.admission?.included !== true) reason = 'SCORE_NOT_ADMITTED';
        else if (!known(m.station?.value)) reason = 'STATION_SUPPORT_UNKNOWN';
        else if (finite(cal.sample_rate_hz) === null || cal.sample_rate_hz <= 0) reason = 'PROCESSING_SUPPORT_UNKNOWN';
        else if (method === 'seismic_thd' && (!known(cal.baseline?.window) || finite(cal.baseline?.mean) === null || finite(cal.baseline?.std) === null)) reason = 'BASELINE_UNKNOWN';
        // Reports do not retain a complete unit/algorithm-version identity. This is
        // a numerical change of the recorded field, NOT a validated sensor derivative.
        // Compare every retained stable field; daily source locators and z-scores
        // are provenance/outcomes, not continuity identities.
        const identity = JSON.stringify(canonical({method, station:m.station, sample_rate_hz:cal.sample_rate_hz,
            baseline:cal.baseline, score_basis:cal.score_basis, calibration:cal.calibration &&
                Object.fromEntries(Object.entries(cal.calibration).filter(([k]) => k !== 'reason')),
            unit:m.raw?.unit ?? m.unit, method_version:m.method_version, processing:m.processing_id,
            regime:m.regime_id, support:m.support, admission:basis === 'score' ? m.admission : null}));
        return {value:finite(value?.value), reason, identity, station:m.station?.value || null,
            unit:m.raw?.unit ?? m.unit ?? null, version:m.method_version ?? null, source:source.sha256 || null};
    }
    function rows(days, method, basis = 'raw', region = '') {
        if (!['raw', 'score'].includes(basis)) throw new Error('RATE_BASIS_INVALID');
        const counts = new Map();
        days.forEach(d => counts.set(d.date, (counts.get(d.date) || 0) + 1));
        let previous = null;
        return days.slice().sort((a,b) => String(a.date).localeCompare(String(b.date))).map(d => {
            const e = endpoint(d, method, basis);
            if (counts.get(d.date) !== 1) e.reason = 'DUPLICATE_REPORT_DAY';
            const r = {date:d.date, previous_date:previous?.date ?? null, region, method, basis,
                current:e.value, previous:previous?.endpoint.value ?? null, elapsed_days:null, change_per_day:null,
                status:'UNKNOWN', reason:e.reason, station:e.station, recorded_unit:e.unit,
                recorded_method_version:e.version, current_source_sha256:e.source,
                previous_source_sha256:previous?.endpoint.source ?? null, clock:'REPORT_DAY_NOT_SENSOR_OR_AVAILABILITY', definition:VERSION};
            if (!r.reason) {
                if (!previous) r.reason = 'NO_PREVIOUS_DAY';
                else if (previous.endpoint.reason) r.reason = 'PREVIOUS_' + previous.endpoint.reason;
                else {
                    r.elapsed_days = (dayTime(d.date) - dayTime(previous.date)) / 86400000;
                    if (r.elapsed_days !== 1) r.reason = 'MISSING_REPORT_DAY';
                    else if (e.identity !== previous.endpoint.identity) r.reason = 'SUPPORT_BASELINE_OR_PROCESSING_CHANGE';
                    else {
                        const change = (e.value - previous.endpoint.value) / r.elapsed_days;
                        if (!Number.isFinite(change)) r.reason = 'NONFINITE_CHANGE';
                        else {r.change_per_day = change; r.status = 'KNOWN';}
                    }
                }
            }
            previous = {date:d.date, endpoint:e};
            return r;
        });
    }
    function describe(r) {
        return r.date + ' | ' + r.region + ' | ' + r.method + ' | ' + r.basis + '\n' +
            (r.status === 'KNOWN' ? '(' + r.current + ' - ' + r.previous + ') / ' + r.elapsed_days + ' report day = ' + r.change_per_day
                : 'UNKNOWN: ' + r.reason) + '\nEndpoints: ' + (r.previous_date || 'none') + ' -> ' + r.date +
            '\nStation: ' + (r.station || 'not retained') + '; unit: ' + (r.recorded_unit || 'not retained') +
            '; method version: ' + (r.recorded_method_version || 'not retained') +
            '\nSource SHA256: ' + (r.previous_source_sha256 || 'none') + ' -> ' + (r.current_source_sha256 || 'none') +
            '\nRecorded-number difference only. No physical derivative, sensor-time or warning claim.';
    }
    function csv(records) {
        const quote = v => '"' + String(v ?? '').replace(/"/g, '""') + '"';
        return COLUMNS.join(',') + '\n' + records.map(r => COLUMNS.map(k => quote(r[k])).join(',')).join('\n') + '\n';
    }
    return {VERSION, COLUMNS, dayTime, rows, describe, csv};
});
