import numpy as np

from ..text_tools import calculate_cer, winsorized_mean


class SampleResult:
    """Minimal container for evaluator inputs."""

    def __init__(self, data_dict):
        self.sample_id = data_dict.get("id", "")
        self.ground_truth = data_dict.get("gt", "")
        self.baseline_pred = data_dict.get("baseline_pred", "")
        self.baseline_meltdown = data_dict.get("baseline_meltdown", False)
        self.system_pred = data_dict.get("system_pred", "")
        self.system_pass = data_dict.get("system_pass", False)
        self.system_meltdown = data_dict.get("system_meltdown", False)


class Evaluator:
    def __init__(self, results, case_sensitive: bool = False, meltdown_t: float = 2.0):
        self.results = results
        self.case_sensitive = case_sensitive
        self.meltdown_t = meltdown_t

    def evaluate(self):
        n = len(self.results)
        if n == 0:
            return {}

        baseline_cers = []
        system_cers_all = []
        system_cers_cov = []
        baseline_cers_on_cov = []

        baseline_melt = 0
        system_melt_all = 0
        system_melt_cov = 0

        baseline_exposure_sum = 0.0
        system_exposure_sum = 0.0
        baseline_exposure_meltdown_count = 0
        system_exposure_meltdown_count = 0

        covered_count = 0
        baseline_melt_and_pass = 0

        for r in self.results:
            cer_b = calculate_cer(r.baseline_pred, r.ground_truth, case_sensitive=self.case_sensitive)
            cer_s = calculate_cer(r.system_pred, r.ground_truth, case_sensitive=self.case_sensitive)

            baseline_cers.append(cer_b)
            system_cers_all.append(cer_s)

            baseline_exposure_sum += cer_b
            if cer_b > self.meltdown_t:
                baseline_exposure_meltdown_count += 1
            if r.baseline_meltdown:
                baseline_melt += 1

            if r.system_meltdown:
                system_melt_all += 1

            if r.system_pass:
                covered_count += 1
                system_cers_cov.append(cer_s)
                baseline_cers_on_cov.append(cer_b)
                system_exposure_sum += cer_s
                if cer_s > self.meltdown_t:
                    system_exposure_meltdown_count += 1
                if r.system_meltdown:
                    system_melt_cov += 1
                if r.baseline_meltdown:
                    baseline_melt_and_pass += 1

        coverage = covered_count / n if n > 0 else 0.0

        baseline_melt_rate = baseline_melt / n if n > 0 else 0.0
        system_melt_rate_all = system_melt_all / n if n > 0 else 0.0
        system_melt_rate_cov = (system_melt_cov / covered_count) if covered_count > 0 else 0.0

        blocked = baseline_melt - baseline_melt_and_pass
        suppression = blocked / baseline_melt if baseline_melt > 0 else 1.0

        baseline_exposure_mean = baseline_exposure_sum / n if n > 0 else 0.0
        system_exposure_mean = system_exposure_sum / n if n > 0 else 0.0

        baseline_exposure_meltdown_rate = baseline_exposure_meltdown_count / n if n > 0 else 0.0
        system_exposure_meltdown_rate = system_exposure_meltdown_count / n if n > 0 else 0.0

        def pct(arr, p):
            return float(np.percentile(np.array(arr, dtype=np.float32), p)) if arr else 0.0

        report = {
            "Samples": n,
            "Baseline_CER_Mean": float(np.mean(baseline_cers)) if baseline_cers else 0.0,
            "Baseline_CER_P95": pct(baseline_cers, 95),
            "Baseline_CER_P99": pct(baseline_cers, 99),
            f"Baseline_ExposureMeltdown_{self.meltdown_t}": baseline_exposure_meltdown_rate,
            "Baseline_Exposure_Mean": baseline_exposure_mean,
            "Baseline_Winsor_Mean_P95": winsorized_mean(baseline_cers, 95),
            "Baseline_Winsor_Mean_P99": winsorized_mean(baseline_cers, 99),
            "Baseline_Meltdown_Rate": baseline_melt_rate,
            "System_Coverage": coverage,
            "System_CER_Mean_All": float(np.mean(system_cers_all)) if system_cers_all else 0.0,
            "System_CER_Mean_Covered": float(np.mean(system_cers_cov)) if system_cers_cov else 0.0,
            "System_CER_P95_Covered": pct(system_cers_cov, 95),
            "System_CER_P99_Covered": pct(system_cers_cov, 99),
            f"System_ExposureMeltdown_{self.meltdown_t}": system_exposure_meltdown_rate,
            "System_Exposure_Mean": system_exposure_mean,
            "System_Meltdown_Rate_All": system_melt_rate_all,
            "System_Meltdown_Rate_Covered": system_melt_rate_cov,
            "Meltdown_Suppression_Rate": float(suppression),
            "Meltdown_Leak_Rate": float(baseline_melt_and_pass / baseline_melt) if baseline_melt > 0 else 0.0,
            "Baseline_CER_Mean_on_SystemCovered": float(np.mean(baseline_cers_on_cov)) if baseline_cers_on_cov else 0.0,
        }
        return report
