(function () {
    'use strict';

    var API_URL = 'https://huggingface.co/api/models?author=kojima-lab&limit=200';
    var ORG_PREFIX = 'kojima-lab/';
    var REPO_PREFIX = 'molcrawl-';

    var SIZE_ORDER = ['small', 'medium', 'large', 'xl'];
    var ARCHS = ['gpt2', 'bert', 'dnabert2', 'esm2', 'chemberta2', 'rnaformer'];
    var ARCH_LABELS = {
        gpt2: 'GPT-2',
        bert: 'BERT',
        dnabert2: 'DNABERT-2',
        esm2: 'ESM-2',
        chemberta2: 'ChemBERTa-2',
        rnaformer: 'RNAformer'
    };

    var BASE_MODALITIES = [
        { key: 'genome-sequence', label: 'Genome Sequence' },
        { key: 'protein-sequence', label: 'Protein Sequence' },
        { key: 'compounds', label: 'Compounds' },
        { key: 'rna', label: 'RNA' },
        { key: 'molecule-nat-lang', label: 'Molecule Natural Language' }
    ];

    var DATASET_LABELS = {
        chembl: 'ChEMBL',
        guacamol: 'GuacaMol',
        clinvar: 'ClinVar',
        proteingym: 'ProteinGym',
        celltype: 'Cell-type',
        'mol-instructions': 'Mol-Instructions'
    };

    var FOUNDATION_ROWS = [
        ['genome-sequence', 'gpt2'],
        ['genome-sequence', 'bert'],
        ['genome-sequence', 'dnabert2'],
        ['protein-sequence', 'gpt2'],
        ['protein-sequence', 'bert'],
        ['protein-sequence', 'esm2'],
        ['compounds', 'gpt2'],
        ['compounds', 'bert'],
        ['compounds', 'chemberta2'],
        ['rna', 'gpt2'],
        ['rna', 'bert'],
        ['rna', 'rnaformer'],
        ['molecule-nat-lang', 'gpt2'],
        ['molecule-nat-lang', 'bert']
    ];

    function parseRepoId(repoId) {
        if (!repoId || repoId.indexOf(ORG_PREFIX + REPO_PREFIX) !== 0) return null;
        var remainder = repoId.slice((ORG_PREFIX + REPO_PREFIX).length);
        var parts = remainder.split('-');

        var size = null;
        if (SIZE_ORDER.indexOf(parts[parts.length - 1]) !== -1) {
            size = parts.pop();
        }

        var arch = null;
        if (parts.length && ARCHS.indexOf(parts[parts.length - 1]) !== -1) {
            arch = parts.pop();
        }

        var modalityKey = parts.join('-');
        var base = null;
        var dataset = null;
        for (var i = 0; i < BASE_MODALITIES.length; i++) {
            var m = BASE_MODALITIES[i];
            if (modalityKey === m.key) { base = m.key; dataset = null; break; }
            if (modalityKey.indexOf(m.key + '-') === 0) {
                base = m.key;
                dataset = modalityKey.slice(m.key.length + 1);
                break;
            }
        }

        return {
            repoId: repoId,
            raw: remainder,
            base: base,
            dataset: dataset,
            arch: arch,
            size: size
        };
    }

    async function fetchModels() {
        var all = [];
        var url = API_URL;
        var safety = 10;
        while (url && safety-- > 0) {
            var res = await fetch(url, { headers: { 'Accept': 'application/json' } });
            if (!res.ok) throw new Error('HTTP ' + res.status);
            var data = await res.json();
            all = all.concat(data);
            var link = res.headers.get('Link');
            var next = null;
            if (link) {
                var m = link.match(/<([^>]+)>;\s*rel="next"/);
                if (m) next = m[1];
            }
            url = next;
        }
        return all;
    }

    function organize(records) {
        var foundation = new Map();
        var finetuned = new Map();
        var other = [];
        var sizesUsed = new Set();
        for (var i = 0; i < records.length; i++) {
            var r = records[i];
            if (!r) continue;
            if (r.size) sizesUsed.add(r.size);
            if (!r.base || !r.arch) {
                other.push(r);
                continue;
            }
            var target = r.dataset ? finetuned : foundation;
            var key = r.base + '|' + (r.dataset || '') + '|' + r.arch;
            if (!target.has(key)) {
                target.set(key, { base: r.base, dataset: r.dataset, arch: r.arch, sizes: new Map() });
            }
            target.get(key).sizes.set(r.size || 'unsized', r.repoId);
        }
        return { foundation: foundation, finetuned: finetuned, other: other, sizesUsed: sizesUsed };
    }

    function modalityLabel(key) {
        for (var i = 0; i < BASE_MODALITIES.length; i++) {
            if (BASE_MODALITIES[i].key === key) return BASE_MODALITIES[i].label;
        }
        return key;
    }

    function archLabel(arch) {
        return ARCH_LABELS[arch] || arch || '(unknown)';
    }

    function datasetLabel(dataset) {
        if (!dataset) return '';
        if (DATASET_LABELS[dataset]) return DATASET_LABELS[dataset];
        return dataset.split('-').map(function (s) {
            return DATASET_LABELS[s] || (s.charAt(0).toUpperCase() + s.slice(1));
        }).join(' ');
    }

    function sizeLabel(size) {
        if (size === 'xl') return 'XL';
        return size.charAt(0).toUpperCase() + size.slice(1);
    }

    function makeEl(tag, opts) {
        var el = document.createElement(tag);
        if (opts) {
            if (opts.className) el.className = opts.className;
            if (opts.text != null) el.textContent = opts.text;
            if (opts.html != null) el.innerHTML = opts.html;
            if (opts.attrs) {
                for (var k in opts.attrs) el.setAttribute(k, opts.attrs[k]);
            }
        }
        return el;
    }

    function buildMatrix(rows, sizes, getLabel, getRecord) {
        var wrapper = makeEl('div', { className: 'model-table-wrapper' });
        var table = makeEl('table', { className: 'model-table model-matrix' });
        var thead = makeEl('thead');
        var headRow = makeEl('tr');
        var kindTh = makeEl('th', { text: 'Model Kind' });
        kindTh.scope = 'col';
        headRow.appendChild(kindTh);
        sizes.forEach(function (s) {
            var th = makeEl('th', { text: sizeLabel(s) });
            th.scope = 'col';
            headRow.appendChild(th);
        });
        thead.appendChild(headRow);
        table.appendChild(thead);

        var tbody = makeEl('tbody');
        rows.forEach(function (row) {
            var tr = makeEl('tr');
            var th = makeEl('th', { text: getLabel(row) });
            th.scope = 'row';
            tr.appendChild(th);
            var rec = getRecord(row);
            sizes.forEach(function (s) {
                var td = makeEl('td');
                var repoId = rec && rec.sizes.get(s);
                if (repoId) {
                    var a = makeEl('a', {
                        text: repoId.slice(ORG_PREFIX.length),
                        attrs: {
                            href: 'https://huggingface.co/' + repoId,
                            target: '_blank',
                            rel: 'noopener'
                        }
                    });
                    td.appendChild(a);
                } else {
                    td.appendChild(makeEl('span', { className: 'cell-empty', text: '—' }));
                }
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        wrapper.appendChild(table);
        return wrapper;
    }

    function renderError(container, message) {
        container.innerHTML = '';
        var p = makeEl('p', { className: 'models-error' });
        p.appendChild(document.createTextNode('Failed to load models from Hugging Face: ' + message + '. See '));
        var a = makeEl('a', {
            text: 'huggingface.co/kojima-lab',
            attrs: { href: 'https://huggingface.co/kojima-lab', target: '_blank', rel: 'noopener' }
        });
        p.appendChild(a);
        p.appendChild(document.createTextNode(' for the full list.'));
        container.appendChild(p);
    }

    async function render() {
        var container = document.getElementById('models-matrix-container');
        if (!container) return;
        try {
            var raw = await fetchModels();
            var records = raw.map(function (m) {
                return parseRepoId(m.id || m.modelId);
            }).filter(function (x) { return x; });

            var organized = organize(records);
            var sizes = SIZE_ORDER.filter(function (s) { return organized.sizesUsed.has(s); });
            if (sizes.length === 0) sizes = SIZE_ORDER.slice(0, 1);

            container.innerHTML = '';

            var hF = makeEl('h3', { className: 'subsection-title', text: 'Foundation Models' });
            container.appendChild(hF);
            container.appendChild(buildMatrix(
                FOUNDATION_ROWS,
                sizes,
                function (row) { return modalityLabel(row[0]) + ' — ' + archLabel(row[1]); },
                function (row) { return organized.foundation.get(row[0] + '||' + row[1]); }
            ));

            if (organized.finetuned.size > 0) {
                var ftRows = Array.from(organized.finetuned.values()).sort(function (a, b) {
                    var ka = a.base + '|' + a.dataset + '|' + a.arch;
                    var kb = b.base + '|' + b.dataset + '|' + b.arch;
                    return ka < kb ? -1 : ka > kb ? 1 : 0;
                });
                var hFT = makeEl('h3', { className: 'subsection-title', text: 'Dataset-specific Fine-tuned Variants' });
                container.appendChild(hFT);
                container.appendChild(buildMatrix(
                    ftRows,
                    sizes,
                    function (rec) {
                        return modalityLabel(rec.base) + ' (' + datasetLabel(rec.dataset) + ') — ' + archLabel(rec.arch);
                    },
                    function (rec) { return rec; }
                ));
            }

            if (organized.other.length > 0) {
                var hO = makeEl('h3', { className: 'subsection-title', text: 'Other Repositories' });
                container.appendChild(hO);
                var ul = makeEl('ul', { className: 'models-list' });
                organized.other.forEach(function (r) {
                    var li = makeEl('li');
                    li.appendChild(makeEl('a', {
                        text: r.repoId,
                        attrs: { href: 'https://huggingface.co/' + r.repoId, target: '_blank', rel: 'noopener' }
                    }));
                    ul.appendChild(li);
                });
                container.appendChild(ul);
            }

            var meta = makeEl('p', { className: 'models-meta' });
            meta.appendChild(document.createTextNode(
                'Loaded ' + records.length + ' kojima-lab/molcrawl-* repositories live from '
            ));
            meta.appendChild(makeEl('a', {
                text: 'huggingface.co/kojima-lab',
                attrs: { href: 'https://huggingface.co/kojima-lab', target: '_blank', rel: 'noopener' }
            }));
            meta.appendChild(document.createTextNode('.'));
            container.appendChild(meta);
        } catch (err) {
            renderError(container, err && err.message ? err.message : String(err));
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', render);
    } else {
        render();
    }
})();
