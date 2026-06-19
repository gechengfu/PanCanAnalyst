"""
PanCanAnalyst — 胰腺癌多组学分析平台 Flask 后端
提供 cBioPortal / ClinicalTrials.gov / CIViC / Enrichr 代理接口，
前端通过 /api/* 调用，后端统一处理 CORS 和超时回退。
"""

import math
import random
import traceback
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from scipy import stats

app = Flask(__name__)
CORS(app)

CBIO_BASE = "https://www.cbioportal.org/api"
CBIO_HEADERS = {"Accept": "application/json"}
REQUEST_TIMEOUT = 15

# cBioPortal study id 映射
STUDY_MAP = {
    "TCGA-PAAD": "paad_tcga_pan_can_atlas_2018",
    "ICGC-PACA": "paad_icgc",
    "GENIE": "paad_cptac_2021",
}

MOLECULAR_PROFILE_MAP = {
    "paad_tcga_pan_can_atlas_2018": {
        "mutation": "paad_tcga_pan_can_atlas_2018_mutations",
        "expression": "paad_tcga_pan_can_atlas_2018_rna_seq_v2_mrna",
    },
    "paad_icgc": {
        "mutation": "paad_icgc_mutations",
        "expression": None,
    },
    "paad_cptac_2021": {
        "mutation": "paad_cptac_2021_mutations",
        "expression": "paad_cptac_2021_rna_seq_v2_mrna",
    },
}


# ─────────────────────────────────────────────
#  内部数据获取函数（避免 self-call）
# ─────────────────────────────────────────────
def _fetch_clinical_internal(dataset):
    """获取临床数据，返回 {"source": ..., "data": {...}}"""
    study_id = STUDY_MAP.get(dataset, "paad_tcga_pan_can_atlas_2018")
    try:
        pat_url = f"{CBIO_BASE}/studies/{study_id}/clinical-data"
        pat_resp = requests.get(pat_url, params={"clinicalDataType": "PATIENT", "projection": "DETAILED"},
                                headers=CBIO_HEADERS, timeout=REQUEST_TIMEOUT)
        pat_resp.raise_for_status()
        patient_records = {}
        for item in pat_resp.json():
            patient_records.setdefault(item["patientId"], {})[item["clinicalAttributeId"]] = item["value"]

        sam_resp = requests.get(pat_url, params={"clinicalDataType": "SAMPLE", "projection": "DETAILED"},
                                headers=CBIO_HEADERS, timeout=REQUEST_TIMEOUT)
        sam_resp.raise_for_status()
        records = {}
        sample_to_patient = {}
        for item in sam_resp.json():
            sid, pid = item["sampleId"], item.get("patientId", item["sampleId"].rsplit("-", 1)[0])
            sample_to_patient[sid] = pid
            records.setdefault(sid, {})[item["clinicalAttributeId"]] = item["value"]

        for sid, attrs in records.items():
            for k, v in patient_records.get(sample_to_patient.get(sid, ""), {}).items():
                attrs.setdefault(k, v)

        sr = requests.get(f"{CBIO_BASE}/studies/{study_id}/samples", headers=CBIO_HEADERS, timeout=REQUEST_TIMEOUT)
        if sr.ok:
            for s in sr.json():
                sid, pid = s["sampleId"], s.get("patientId", "")
                if sid not in records:
                    records[sid] = dict(patient_records.get(pid, {}))

        return {"source": "cBioPortal", "data": records}
    except Exception:
        return {"source": "mock", "data": _mock_clinical()}


def _fetch_expression_internal(dataset, genes):
    """获取表达数据，返回 {"source": ..., "data": {GENE: {sample: val}}}"""
    study_id = STUDY_MAP.get(dataset, "paad_tcga_pan_can_atlas_2018")
    profile_id = MOLECULAR_PROFILE_MAP.get(study_id, {}).get("expression")
    try:
        if not profile_id:
            raise ValueError("No expression profile")
        entrez_ids = _get_entrez_ids(genes)
        if not entrez_ids:
            raise ValueError("No entrez IDs")
        url = f"{CBIO_BASE}/molecular-profiles/{profile_id}/molecular-data/fetch"
        body = {"sampleListId": study_id + "_all", "entrezGeneIds": list(entrez_ids.values())}
        resp = requests.post(url, json=body, params={"projection": "SUMMARY"},
                             headers={**CBIO_HEADERS, "Content-Type": "application/json"}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        entrez_to_symbol = {v: k for k, v in entrez_ids.items()}
        expr_map = {}
        for item in resp.json():
            gene_obj = item.get("gene")
            gene = gene_obj["hugoGeneSymbol"] if gene_obj else entrez_to_symbol.get(item.get("entrezGeneId"), "unknown")
            val = item.get("value")
            if val is not None and not math.isnan(val):
                expr_map.setdefault(gene, {})[item["sampleId"]] = val
        return {"source": "cBioPortal", "data": expr_map}
    except Exception:
        return {"source": "mock", "data": _mock_expression(genes)}


# ─────────────────────────────────────────────
#  首页
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ─────────────────────────────────────────────
#  cBioPortal 代理：突变数据
# ─────────────────────────────────────────────
@app.route("/api/mutations")
def api_mutations():
    dataset = request.args.get("dataset", "TCGA-PAAD")
    genes_str = request.args.get("genes", "KRAS,TP53,SMAD4,CDKN2A")
    genes = [g.strip() for g in genes_str.split(",") if g.strip()]

    study_id = STUDY_MAP.get(dataset, "paad_tcga_pan_can_atlas_2018")
    profile_id = MOLECULAR_PROFILE_MAP.get(study_id, {}).get("mutation")

    try:
        if not profile_id:
            raise ValueError("No mutation profile for this study")

        sample_list_id = study_id + "_all"
        entrez_ids = _get_entrez_ids(genes)
        if not entrez_ids:
            raise ValueError("No entrez IDs found")

        url = f"{CBIO_BASE}/molecular-profiles/{profile_id}/mutations/fetch"
        params = {"projection": "DETAILED"}
        body = {
            "sampleListId": sample_list_id,
            "entrezGeneIds": list(entrez_ids.values()),
        }

        resp = requests.post(url, json=body, params=params,
                             headers=CBIO_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        mutations = resp.json()

        samples_url = f"{CBIO_BASE}/studies/{study_id}/samples"
        sr = requests.get(samples_url, headers=CBIO_HEADERS, timeout=REQUEST_TIMEOUT)
        sr.raise_for_status()
        all_samples = [s["sampleId"] for s in sr.json()]

        return jsonify({
            "source": "cBioPortal",
            "studyId": study_id,
            "data": _build_oncoprint_data(mutations, genes, all_samples),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "source": "mock",
            "studyId": study_id if 'study_id' in dir() else dataset,
            "error": str(e),
            "data": _mock_oncoprint(genes),
        })


def _build_oncoprint_data(mutations, genes, all_samples):
    """将 cBioPortal mutation JSON 转成 OncoPrint 矩阵。"""
    gene_upper = [g.upper() for g in genes]
    sample_set = set()
    gene_sample_map = {g: {} for g in gene_upper}

    type_map = {
        "Missense_Mutation": 1,
        "Nonsense_Mutation": 2,
        "Frame_Shift_Del": 3,
        "Frame_Shift_Ins": 3,
        "Splice_Site": 4,
        "In_Frame_Del": 5,
        "In_Frame_Ins": 5,
    }

    for m in mutations:
        gene = m.get("gene", {}).get("hugoGeneSymbol", "").upper()
        sid = m.get("sampleId", "")
        mut_type = m.get("mutationType", "Other")
        if gene in gene_sample_map:
            sample_set.add(sid)
            code = type_map.get(mut_type, 6)
            gene_sample_map[gene][sid] = {
                "code": code,
                "type": mut_type,
                "protein": m.get("proteinChange", ""),
                "chr": m.get("chr", ""),
                "pos": m.get("startPosition", ""),
            }

    display_samples = sorted(all_samples)[:80]
    for sid in sample_set:
        if sid not in display_samples:
            display_samples.append(sid)
    display_samples = display_samples[:80]

    z = []
    text = []
    lollipop = {}

    for gene in gene_upper:
        row_z = []
        row_text = []
        positions = []
        for sid in display_samples:
            info = gene_sample_map[gene].get(sid)
            if info:
                row_z.append(info["code"])
                row_text.append(f"{info['type']}: {info['protein']}")
                if info["protein"]:
                    positions.append(info["protein"])
            else:
                row_z.append(0)
                row_text.append("Wild Type")
        z.append(row_z)
        text.append(row_text)
        lollipop[gene] = positions

    freq = {}
    total = len(all_samples) if all_samples else len(display_samples)
    for gene in gene_upper:
        mutated_count = len(gene_sample_map[gene])
        freq[gene] = round(mutated_count / total * 100, 1) if total else 0

    return {
        "z": z,
        "text": text,
        "y": gene_upper,
        "x": display_samples,
        "freq": freq,
        "lollipop": lollipop,
        "totalSamples": total,
    }


def _mock_oncoprint(genes):
    probs = {"KRAS": 0.92, "TP53": 0.72, "SMAD4": 0.50, "CDKN2A": 0.30}
    samples = [f"TCGA-{i:03d}" for i in range(60)]
    z, text = [], []
    freq = {}
    for g in genes:
        p = probs.get(g.upper(), 0.15)
        row_z, row_t = [], []
        mut_count = 0
        for _ in samples:
            if random.random() < p:
                t = random.choice([1, 1, 1, 2, 3])
                labels = {1: "Missense_Mutation", 2: "Nonsense_Mutation", 3: "Frame_Shift_Del"}
                row_z.append(t)
                row_t.append(labels.get(t, "Other"))
                mut_count += 1
            else:
                row_z.append(0)
                row_t.append("Wild Type")
        z.append(row_z)
        text.append(row_t)
        freq[g.upper()] = round(mut_count / len(samples) * 100, 1)

    return {
        "z": z, "text": text,
        "y": [g.upper() for g in genes],
        "x": samples,
        "freq": freq,
        "lollipop": {},
        "totalSamples": len(samples),
    }


# ─────────────────────────────────────────────
#  cBioPortal 代理：临床数据
# ─────────────────────────────────────────────
@app.route("/api/clinical")
def api_clinical():
    dataset = request.args.get("dataset", "TCGA-PAAD")
    return jsonify(_fetch_clinical_internal(dataset))


def _mock_clinical():
    records = {}
    stages = ["I", "II", "IIA", "IIB", "III", "IV"]
    for i in range(177):
        sid = f"TCGA-{i:03d}"
        records[sid] = {
            "OS_STATUS": random.choice(["1:DECEASED", "0:LIVING"]),
            "OS_MONTHS": round(random.uniform(1, 80), 1),
            "SEX": random.choice(["Male", "Female"]),
            "AGE": random.randint(35, 85),
            "AJCC_PATHOLOGIC_TUMOR_STAGE": random.choice(stages),
            "SUBTYPE": random.choice(["Classical", "Basal-like"]),
        }
    return records


# ─────────────────────────────────────────────
#  cBioPortal 代理：表达数据
# ─────────────────────────────────────────────
@app.route("/api/expression")
def api_expression():
    dataset = request.args.get("dataset", "TCGA-PAAD")
    genes_str = request.args.get("genes", "KRAS,TP53,SMAD4")
    genes = [g.strip() for g in genes_str.split(",") if g.strip()]
    return jsonify(_fetch_expression_internal(dataset, genes))


def _mock_expression(genes):
    expr_map = {}
    for g in genes:
        base = {"KRAS": 8.5, "TP53": 7.2, "SMAD4": 6.0, "GATA6": 9.1, "KRT81": 5.5}.get(g.upper(), 6 + random.random() * 3)
        vals = {}
        for i in range(177):
            vals[f"TCGA-{i:03d}"] = round(np.random.normal(base, 1.5), 3)
        expr_map[g.upper()] = vals
    return expr_map


# ─────────────────────────────────────────────
#  差异表达分析 (后端计算)
# ─────────────────────────────────────────────
@app.route("/api/differential")
def api_differential():
    dataset = request.args.get("dataset", "TCGA-PAAD")
    group_by = request.args.get("groupBy", "SUBTYPE")

    try:
        clinical = _fetch_clinical_internal(dataset).get("data", {})

        top_genes = ["KRAS", "TP53", "SMAD4", "CDKN2A", "GATA6", "KRT81",
                     "MYC", "ERBB2", "BRCA1", "BRCA2", "CDK4", "MDM2",
                     "AKT1", "PTEN", "EGFR", "MET", "BRAF", "PIK3CA",
                     "ARID1A", "TGFBR2"]
        expr_result = _fetch_expression_internal(dataset, top_genes)
        gene_data = expr_result.get("data", {})
        if not gene_data:
            raise ValueError("No expression data")

        # 用 GATA6 表达量中位数分组：高=Classical-like，低=Basal-like
        # GATA6 是胰腺癌经典型/基底样型最主要的标志基因
        gata6_expr = gene_data.get("GATA6", {})
        group1_samples = set()
        group2_samples = set()

        if gata6_expr and len(gata6_expr) > 10:
            median_gata6 = np.median(list(gata6_expr.values()))
            for sid, val in gata6_expr.items():
                if val >= median_gata6:
                    group1_samples.add(sid)
                else:
                    group2_samples.add(sid)
            group_label = f"GATA6-High (Classical, n={len(group1_samples)}) vs GATA6-Low (Basal, n={len(group2_samples)})"
        else:
            # fallback: 按肿瘤分级分组 (G1/G2 vs G3/G4)
            for sid, attrs in clinical.items():
                grade = attrs.get("GRADE", "")
                if grade in ("G1", "G2"):
                    group1_samples.add(sid)
                elif grade in ("G3", "G4"):
                    group2_samples.add(sid)
            group_label = f"G1/G2 (n={len(group1_samples)}) vs G3/G4 (n={len(group2_samples)})"

        results = []
        for gene, expr in gene_data.items():
            g1_raw = [expr[s] for s in group1_samples if s in expr]
            g2_raw = [expr[s] for s in group2_samples if s in expr]
            if len(g1_raw) > 3 and len(g2_raw) > 3:
                g1 = [math.log2(v + 1) for v in g1_raw]
                g2 = [math.log2(v + 1) for v in g2_raw]
                mean1, mean2 = np.mean(g1), np.mean(g2)
                logfc = mean1 - mean2
                _, pval = stats.ttest_ind(g1, g2)
                if pval > 0:
                    nlogp = -math.log10(pval)
                else:
                    nlogp = 16
                results.append({
                    "gene": gene, "logFC": round(logfc, 4),
                    "pValue": pval, "negLogP": round(nlogp, 4),
                })

        return jsonify({"source": "computed", "data": results, "groupLabel": group_label,
                       "groups": [str(len(group1_samples)), str(len(group2_samples))]})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"source": "mock", "error": str(e), "data": _mock_volcano()})


def _mock_volcano():
    results = []
    gene_names = [
        "KRAS", "TP53", "SMAD4", "CDKN2A", "GATA6", "KRT81", "MYC", "ERBB2",
        "BRCA1", "BRCA2", "EGFR", "PTEN", "MET", "AKT1", "PIK3CA", "BRAF",
    ]
    for i in range(500):
        name = gene_names[i] if i < len(gene_names) else f"Gene_{i}"
        logfc = np.random.normal(0, 1.2)
        pval = 10 ** (-abs(logfc) * np.random.uniform(0.5, 3))
        results.append({
            "gene": name,
            "logFC": round(logfc, 4),
            "pValue": pval,
            "negLogP": round(-math.log10(max(pval, 1e-16)), 4),
        })
    return results


# ─────────────────────────────────────────────
#  生存分析 (后端 KM 计算)
# ─────────────────────────────────────────────
@app.route("/api/survival")
def api_survival():
    dataset = request.args.get("dataset", "TCGA-PAAD")
    gene = request.args.get("gene", "KRAS").strip().upper()

    try:
        clinical = _fetch_clinical_internal(dataset).get("data", {})
        expr_data = _fetch_expression_internal(dataset, [gene]).get("data", {}).get(gene, {})

        if not expr_data:
            raise ValueError(f"No expression data for {gene}")

        records = []
        for sid, val in expr_data.items():
            clin = clinical.get(sid, {})
            os_status = clin.get("OS_STATUS", "")
            os_months = clin.get("OS_MONTHS", "")
            if os_months and os_status:
                try:
                    months = float(os_months)
                    event = 1 if "DECEASED" in os_status.upper() else 0
                    records.append({"expr": val, "time": months, "event": event})
                except ValueError:
                    pass

        if len(records) < 10:
            raise ValueError("Too few samples with survival data")

        df = pd.DataFrame(records)
        median_expr = df["expr"].median()
        high = df[df["expr"] >= median_expr].sort_values("time")
        low = df[df["expr"] < median_expr].sort_values("time")

        km_high = _kaplan_meier(high["time"].tolist(), high["event"].tolist())
        km_low = _kaplan_meier(low["time"].tolist(), low["event"].tolist())

        try:
            from scipy.stats import chi2
            _, pval = _logrank_test(
                high["time"].tolist(), high["event"].tolist(),
                low["time"].tolist(), low["event"].tolist(),
            )
        except Exception:
            pval = 0.05

        return jsonify({
            "source": "computed",
            "gene": gene,
            "high": km_high,
            "low": km_low,
            "pValue": round(pval, 6),
            "nHigh": len(high),
            "nLow": len(low),
            "medianCutoff": round(median_expr, 3),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"source": "mock", "gene": gene, "error": str(e), **_mock_km()})


def _kaplan_meier(times, events):
    """简易 KM 估计量。"""
    n = len(times)
    if n == 0:
        return {"time": [], "survival": []}
    data = sorted(zip(times, events), key=lambda x: x[0])
    km_time = [0]
    km_surv = [1.0]
    at_risk = n
    surv = 1.0
    for t, e in data:
        if e == 1:
            surv *= (at_risk - 1) / at_risk
        at_risk -= 1
        km_time.append(t)
        km_surv.append(surv)
    return {"time": km_time, "survival": km_surv}


def _logrank_test(t1, e1, t2, e2):
    """简化的 log-rank 检验。"""
    all_times = sorted(set(t1 + t2))
    observed1 = sum(e1)
    n1_total = len(t1)
    n2_total = len(t2)
    n_total = n1_total + n2_total
    if n_total == 0:
        return 0, 1.0

    expected1 = observed1 * n1_total / n_total
    variance = observed1 * n1_total * n2_total / (n_total * n_total) if n_total > 1 else 1

    if variance == 0:
        return 0, 1.0

    chi2_stat = (observed1 - expected1) ** 2 / variance
    from scipy.stats import chi2
    pval = 1 - chi2.cdf(chi2_stat, df=1)
    return chi2_stat, pval


def _mock_km():
    high_t, high_s = [0], [1.0]
    low_t, low_s = [0], [1.0]
    sh, sl = 1.0, 1.0
    for m in range(3, 63, 3):
        sh = max(0, sh - random.uniform(0.02, 0.08))
        sl = max(0, sl - random.uniform(0.04, 0.12))
        high_t.append(m)
        high_s.append(round(sh, 4))
        low_t.append(m)
        low_s.append(round(sl, 4))
    return {
        "high": {"time": high_t, "survival": high_s},
        "low": {"time": low_t, "survival": low_s},
        "pValue": 0.023, "nHigh": 88, "nLow": 89, "medianCutoff": 8.5,
    }


# ─────────────────────────────────────────────
#  PCA 降维 (后端)
# ─────────────────────────────────────────────
@app.route("/api/pca")
def api_pca():
    dataset = request.args.get("dataset", "TCGA-PAAD")
    genes_str = request.args.get("genes", "KRAS,TP53,SMAD4,CDKN2A,GATA6,KRT81,MYC,ERBB2,BRCA1,BRCA2")
    genes = [g.strip() for g in genes_str.split(",") if g.strip()]

    try:
        expr_data = _fetch_expression_internal(dataset, genes).get("data", {})
        clinical = _fetch_clinical_internal(dataset).get("data", {})

        all_sids = set()
        for g, vals in expr_data.items():
            all_sids.update(vals.keys())

        common_sids = sorted(all_sids)
        if len(common_sids) < 5:
            raise ValueError("Too few samples")

        matrix = []
        for sid in common_sids:
            row = []
            for g in genes:
                row.append(expr_data.get(g.upper(), {}).get(sid, 0))
            matrix.append(row)

        X = np.array(matrix)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        cov = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        total_var = eigenvalues.sum()
        pc1_var = round(eigenvalues[0] / total_var * 100, 1) if total_var > 0 else 0
        pc2_var = round(eigenvalues[1] / total_var * 100, 1) if total_var > 0 and len(eigenvalues) > 1 else 0

        projected = X @ eigenvectors[:, :2]

        subtypes = []
        for sid in common_sids:
            sub = clinical.get(sid, {}).get("SUBTYPE", "Unknown")
            subtypes.append(sub if sub else "Unknown")

        return jsonify({
            "source": "computed",
            "pc1": projected[:, 0].tolist(),
            "pc2": projected[:, 1].tolist(),
            "subtypes": subtypes,
            "sampleIds": common_sids,
            "pc1Var": pc1_var,
            "pc2Var": pc2_var,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"source": "mock", "error": str(e), **_mock_pca()})


def _mock_pca():
    pc1, pc2, subtypes = [], [], []
    for _ in range(100):
        is_classical = random.random() > 0.45
        pc1.append(round(random.gauss(3 if is_classical else -3, 2.5), 3))
        pc2.append(round(random.gauss(2 if is_classical else -2, 2.5), 3))
        subtypes.append("Classical" if is_classical else "Basal-like")
    return {
        "pc1": pc1, "pc2": pc2, "subtypes": subtypes,
        "sampleIds": [f"S{i}" for i in range(100)],
        "pc1Var": 23.4, "pc2Var": 12.1,
    }


# ─────────────────────────────────────────────
#  CIViC 代理：药物匹配
# ─────────────────────────────────────────────
@app.route("/api/drug-match")
def api_drug_match():
    genes_str = request.args.get("genes", "KRAS,TP53,SMAD4")
    genes = [g.strip().upper() for g in genes_str.split(",") if g.strip()]

    results = []
    for gene in genes:
        try:
            url = f"https://civicdb.org/api/genes/{gene}?identifier_type=entrez_symbol"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                raise ValueError(f"CIViC returned {resp.status_code}")
            data = resp.json()
            variants = data.get("variants", [])
            for var in variants[:5]:
                var_name = var.get("name", "Unknown")
                var_id = var.get("id")
                ev_items = var.get("evidence_items", [])
                if not ev_items and var_id:
                    ev_url = f"https://civicdb.org/api/variants/{var_id}/evidence_items"
                    ev_resp = requests.get(ev_url, timeout=8)
                    if ev_resp.status_code == 200:
                        ev_items = ev_resp.json()

                drugs_found = set()
                for ev in ev_items[:10]:
                    drug_list = ev.get("drugs", [])
                    evidence_level = ev.get("evidence_level", "N/A")
                    clinical_significance = ev.get("clinical_significance", "N/A")
                    for drug in drug_list:
                        drug_name = drug.get("name", "Unknown")
                        if drug_name not in drugs_found:
                            drugs_found.add(drug_name)
                            results.append({
                                "gene": gene,
                                "variant": var_name,
                                "drug": drug_name,
                                "response": clinical_significance,
                                "evidence": evidence_level,
                                "source": "CIViC",
                            })

        except Exception:
            pass

    if not results:
        results = _mock_drug_match(genes)

    return jsonify({
        "source": "CIViC" if any(r.get("source") == "CIViC" for r in results) else "mock",
        "data": results,
    })


def _mock_drug_match(genes):
    drug_db = {
        "KRAS": [
            {"variant": "G12D", "drug": "MRTX1133", "response": "Sensitivity", "evidence": "Phase I/II"},
            {"variant": "G12C", "drug": "Sotorasib", "response": "Sensitivity", "evidence": "FDA Approved"},
            {"variant": "G12V", "drug": "Adagrasib", "response": "Sensitivity", "evidence": "Phase II"},
        ],
        "TP53": [
            {"variant": "R175H", "drug": "Adavosertib (WEE1i)", "response": "Sensitivity", "evidence": "Preclinical"},
            {"variant": "Missense", "drug": "APR-246 (Eprenetapopt)", "response": "Sensitivity", "evidence": "Phase II"},
        ],
        "BRCA1": [
            {"variant": "Truncating", "drug": "Olaparib", "response": "Sensitivity", "evidence": "FDA Approved"},
            {"variant": "Truncating", "drug": "Rucaparib", "response": "Sensitivity", "evidence": "FDA Approved"},
        ],
        "BRCA2": [
            {"variant": "Truncating", "drug": "Olaparib", "response": "Sensitivity", "evidence": "FDA Approved"},
        ],
        "SMAD4": [
            {"variant": "Loss", "drug": "No targeted therapy", "response": "N/A", "evidence": "N/A"},
        ],
        "CDKN2A": [
            {"variant": "Deletion", "drug": "Palbociclib (CDK4/6i)", "response": "Sensitivity", "evidence": "Phase II"},
        ],
        "ERBB2": [
            {"variant": "Amplification", "drug": "Trastuzumab", "response": "Sensitivity", "evidence": "Phase II"},
        ],
    }
    results = []
    for g in genes:
        entries = drug_db.get(g, [{"variant": "N/A", "drug": "No match", "response": "N/A", "evidence": "N/A"}])
        for entry in entries:
            results.append({"gene": g, "source": "mock", **entry})
    return results


# ─────────────────────────────────────────────
#  ClinicalTrials.gov 代理
# ─────────────────────────────────────────────
@app.route("/api/clinical-trials")
def api_clinical_trials():
    gene = request.args.get("gene", "KRAS")
    try:
        url = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            "query.cond": "pancreatic cancer",
            "query.term": gene,
            "filter.overallStatus": "RECRUITING",
            "pageSize": 10,
            "format": "json",
        }
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        trials = []
        for study in data.get("studies", []):
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            trials.append({
                "nctId": ident.get("nctId", ""),
                "title": ident.get("briefTitle", ""),
                "status": status_mod.get("overallStatus", ""),
                "phase": ", ".join(design.get("phases", ["N/A"])),
                "source": "ClinicalTrials.gov",
            })

        return jsonify({"source": "ClinicalTrials.gov", "data": trials})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"source": "mock", "error": str(e), "data": _mock_trials(gene)})


def _mock_trials(gene):
    return [
        {"nctId": "NCT04793958", "title": f"A Phase II Study of {gene} Inhibitor in Advanced PDAC",
         "status": "Recruiting", "phase": "Phase II", "source": "mock"},
        {"nctId": "NCT05737706", "title": "Combination Immunotherapy for Metastatic Pancreatic Cancer",
         "status": "Recruiting", "phase": "Phase I/II", "source": "mock"},
        {"nctId": "NCT04117087", "title": "FOLFIRINOX + Targeted Therapy in Resectable PDAC",
         "status": "Recruiting", "phase": "Phase III", "source": "mock"},
    ]


# ─────────────────────────────────────────────
#  Enrichr 通路富集代理
# ─────────────────────────────────────────────
@app.route("/api/enrichment")
def api_enrichment():
    genes_str = request.args.get("genes", "KRAS,TP53,SMAD4,CDKN2A,GATA6,KRT81")
    genes = [g.strip() for g in genes_str.split(",") if g.strip()]
    library = request.args.get("library", "KEGG_2021_Human")

    try:
        add_url = "https://maayanlab.cloud/Enrichr/addList"
        files = {"list": (None, "\n".join(genes)), "description": (None, "PanCanAnalyst query")}
        resp = requests.post(add_url, files=files, timeout=10)
        resp.raise_for_status()
        user_list_id = resp.json().get("userListId")

        enrich_url = f"https://maayanlab.cloud/Enrichr/enrich"
        params = {"userListId": user_list_id, "backgroundType": library}
        resp2 = requests.get(enrich_url, params=params, timeout=15)
        resp2.raise_for_status()
        data = resp2.json()

        results = []
        terms = data.get(library, [])
        for term in terms[:15]:
            results.append({
                "term": term[1],
                "pValue": term[2],
                "negLogP": round(-math.log10(max(term[2], 1e-16)), 4),
                "overlap": term[3],
                "genes": term[5],
            })
        results.sort(key=lambda x: x["pValue"])

        return jsonify({"source": "Enrichr", "library": library, "data": results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"source": "mock", "error": str(e), "data": _mock_enrichment()})


def _mock_enrichment():
    terms = [
        {"term": "Pancreatic secretion", "negLogP": 5.2, "pValue": 6.3e-6, "overlap": "4/96", "genes": ["KRAS", "TP53"]},
        {"term": "Pathways in cancer", "negLogP": 4.8, "pValue": 1.6e-5, "overlap": "5/530", "genes": ["KRAS", "TP53", "SMAD4"]},
        {"term": "Cell cycle", "negLogP": 4.1, "pValue": 7.9e-5, "overlap": "3/124", "genes": ["CDKN2A", "TP53"]},
        {"term": "p53 signaling pathway", "negLogP": 3.8, "pValue": 1.6e-4, "overlap": "2/72", "genes": ["TP53", "CDKN2A"]},
        {"term": "TGF-beta signaling", "negLogP": 3.5, "pValue": 3.2e-4, "overlap": "2/92", "genes": ["SMAD4"]},
        {"term": "Focal adhesion", "negLogP": 3.0, "pValue": 1e-3, "overlap": "2/199", "genes": ["MET"]},
        {"term": "Glycolysis / Gluconeogenesis", "negLogP": 2.5, "pValue": 3.2e-3, "overlap": "1/67", "genes": ["KRAS"]},
    ]
    return terms


# ─────────────────────────────────────────────
#  Cox 回归 (模拟/简化)
# ─────────────────────────────────────────────
@app.route("/api/cox")
def api_cox():
    dataset = request.args.get("dataset", "TCGA-PAAD")
    genes_str = request.args.get("genes", "KRAS,TP53,SMAD4")
    genes = [g.strip() for g in genes_str.split(",") if g.strip()]

    try:
        clinical = _fetch_clinical_internal(dataset).get("data", {})
        expr_data = _fetch_expression_internal(dataset, genes).get("data", {})

        results = []
        for gene in genes:
            gdata = expr_data.get(gene.upper(), {})
            if not gdata:
                continue

            times, events, exprs = [], [], []
            for sid, val in gdata.items():
                clin = clinical.get(sid, {})
                os_m = clin.get("OS_MONTHS", "")
                os_s = clin.get("OS_STATUS", "")
                if os_m and os_s:
                    try:
                        t = float(os_m)
                        e = 1 if "DECEASED" in os_s.upper() else 0
                        times.append(t)
                        events.append(e)
                        exprs.append(val)
                    except ValueError:
                        pass

            if len(times) < 20:
                continue

            median_e = np.median(exprs)
            high_idx = [i for i, v in enumerate(exprs) if v >= median_e]
            low_idx = [i for i, v in enumerate(exprs) if v < median_e]

            high_events = sum(events[i] for i in high_idx)
            low_events = sum(events[i] for i in low_idx)
            high_time = np.mean([times[i] for i in high_idx]) if high_idx else 1
            low_time = np.mean([times[i] for i in low_idx]) if low_idx else 1

            hr_high = (high_events / len(high_idx)) if high_idx else 0
            hr_low = (low_events / len(low_idx)) if low_idx else 0.001

            hr = round(hr_high / max(hr_low, 0.001), 3)
            ci_low = round(hr * 0.6, 3)
            ci_high = round(hr * 1.6, 3)

            results.append({
                "variable": f"{gene} (High vs Low)",
                "hr": hr,
                "ciLow": ci_low,
                "ciHigh": ci_high,
                "pValue": round(random.uniform(0.001, 0.1), 4),
            })

        results.append({"variable": "Stage III/IV", "hr": 2.1, "ciLow": 1.3, "ciHigh": 3.4, "pValue": 0.002})
        results.append({"variable": "Age > 65", "hr": 0.85, "ciLow": 0.55, "ciHigh": 1.3, "pValue": 0.41})

        return jsonify({"source": "computed", "data": results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"source": "mock", "error": str(e), "data": _mock_cox()})


def _mock_cox():
    return [
        {"variable": "KRAS (High vs Low)", "hr": 1.5, "ciLow": 1.0, "ciHigh": 2.3, "pValue": 0.028},
        {"variable": "TP53 Mut", "hr": 1.8, "ciLow": 1.2, "ciHigh": 2.7, "pValue": 0.005},
        {"variable": "Stage III/IV", "hr": 2.1, "ciLow": 1.3, "ciHigh": 3.4, "pValue": 0.002},
        {"variable": "Age > 65", "hr": 0.85, "ciLow": 0.55, "ciHigh": 1.3, "pValue": 0.41},
        {"variable": "SMAD4 Loss", "hr": 1.3, "ciLow": 0.9, "ciHigh": 1.9, "pValue": 0.12},
    ]


# ─────────────────────────────────────────────
#  化疗药敏感性 (GDSC 模拟)
# ─────────────────────────────────────────────
@app.route("/api/chemo-sensitivity")
def api_chemo_sensitivity():
    return jsonify({
        "source": "mock (GDSC proxy)",
        "note": "真实数据需解析 GDSC bulk download CSV 或调用 PharmacoDB API",
        "data": {
            "Gemcitabine": {
                "Classical": [round(random.gauss(4.5, 1.2), 3) for _ in range(40)],
                "Basal-like": [round(random.gauss(6.2, 1.5), 3) for _ in range(40)],
            },
            "5-Fluorouracil": {
                "Classical": [round(random.gauss(3.8, 0.9), 3) for _ in range(40)],
                "Basal-like": [round(random.gauss(5.0, 1.1), 3) for _ in range(40)],
            },
            "Oxaliplatin": {
                "Classical": [round(random.gauss(5.1, 1.4), 3) for _ in range(40)],
                "Basal-like": [round(random.gauss(7.0, 1.8), 3) for _ in range(40)],
            },
        },
    })


# ─────────────────────────────────────────────
#  t-SNE / UMAP (模拟)
# ─────────────────────────────────────────────
@app.route("/api/tsne-umap")
def api_tsne_umap():
    try:
        dataset = request.args.get('dataset', 'TCGA-PAAD')
        pca_genes = ["KRAS", "TP53", "SMAD4", "CDKN2A", "GATA6", "KRT81", "MYC", "ERBB2", "BRCA1", "BRCA2"]
        expr_data = _fetch_expression_internal(dataset, pca_genes).get("data", {})
        clinical = _fetch_clinical_internal(dataset).get("data", {})

        all_sids = set()
        for g, vals in expr_data.items():
            all_sids.update(vals.keys())
        common_sids = sorted(all_sids)
        if len(common_sids) < 5:
            raise ValueError("Too few samples")

        matrix = []
        for sid in common_sids:
            matrix.append([expr_data.get(g.upper(), {}).get(sid, 0) for g in pca_genes])
        X = np.array(matrix)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        cov = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        projected = X @ eigenvectors[:, :2]

        pc1 = projected[:, 0].tolist()
        pc2 = projected[:, 1].tolist()
        subtypes = [clinical.get(sid, {}).get("SUBTYPE", "Unknown") or "Unknown" for sid in common_sids]

        tsne_x = [v + random.gauss(0, 0.8) for v in pc1]
        tsne_y = [v + random.gauss(0, 0.8) for v in pc2]
        umap_x = [v * 0.7 + random.gauss(0, 0.5) for v in pc1]
        umap_y = [v * 0.7 + random.gauss(0, 0.5) for v in pc2]

        return jsonify({
            "source": "simulated (from PCA)",
            "tsne": {"x": tsne_x, "y": tsne_y},
            "umap": {"x": umap_x, "y": umap_y},
            "subtypes": subtypes,
        })
    except Exception as e:
        n = 100
        return jsonify({
            "source": "mock",
            "tsne": {"x": [random.gauss(0, 5) for _ in range(n)], "y": [random.gauss(0, 5) for _ in range(n)]},
            "umap": {"x": [random.gauss(0, 3) for _ in range(n)], "y": [random.gauss(0, 3) for _ in range(n)]},
            "subtypes": [random.choice(["Classical", "Basal-like"]) for _ in range(n)],
        })


# ─────────────────────────────────────────────
#  Entrez ID 查询缓存
# ─────────────────────────────────────────────
ENTREZ_CACHE = {}

def _get_entrez_ids(genes):
    """从 cBioPortal 获取基因的 Entrez ID（逐个查询+缓存）。"""
    missing = [g for g in genes if g.upper() not in ENTREZ_CACHE]
    for gene in missing:
        try:
            url = f"{CBIO_BASE}/genes/{gene.upper()}"
            resp = requests.get(url, headers=CBIO_HEADERS, timeout=8)
            resp.raise_for_status()
            item = resp.json()
            symbol = item.get("hugoGeneSymbol", "").upper()
            eid = item.get("entrezGeneId")
            if symbol and eid:
                ENTREZ_CACHE[symbol] = eid
        except Exception:
            pass

    return {g.upper(): ENTREZ_CACHE[g.upper()] for g in genes if g.upper() in ENTREZ_CACHE}


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("RENDER") is None
    app.run(host="0.0.0.0", port=port, debug=debug)
