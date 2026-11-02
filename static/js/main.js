/* global Chart */
/**
 * main.js
 * Frontend interactivity for Used Car Price Prediction Application.
 * Includes dynamic presets, client-side validation, AJAX prediction handling,
 * and Chart.js visualization initialization.
 */

document.addEventListener("DOMContentLoaded", () => {
    initPresets();
    initPredictionForm();
    initDashboardCharts();
});

/* ==========================================================================
   1. Car Presets for Instant 1-Click Testing
   ========================================================================== */
const PRESETS = {
    budget: {
        brand: "Maruti",
        year: 2019,
        km_driven: 42000,
        fuel: "Petrol",
        transmission: "Manual",
        owner: "First Owner",
        seller_type: "Individual",
        mileage: 22.5,
        engine: 1197,
        max_power: 82.0,
        seats: 5
    },
    suv: {
        brand: "Hyundai",
        year: 2021,
        km_driven: 32000,
        fuel: "Diesel",
        transmission: "Manual",
        owner: "First Owner",
        seller_type: "Dealer",
        mileage: 18.5,
        engine: 1493,
        max_power: 113.4,
        seats: 5
    },
    sedan: {
        brand: "Honda",
        year: 2020,
        km_driven: 38000,
        fuel: "Petrol",
        transmission: "Automatic",
        owner: "First Owner",
        seller_type: "Dealer",
        mileage: 18.0,
        engine: 1498,
        max_power: 119.3,
        seats: 5
    },
    luxury: {
        brand: "BMW",
        year: 2020,
        km_driven: 28000,
        fuel: "Diesel",
        transmission: "Automatic",
        owner: "First Owner",
        seller_type: "Trustmark Dealer",
        mileage: 16.0,
        engine: 1995,
        max_power: 190.0,
        seats: 5
    }
};

function initPresets() {
    const presetButtons = document.querySelectorAll(".preset-btn");
    if (!presetButtons.length) return;

    presetButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const key = btn.getAttribute("data-preset");
            const data = PRESETS[key];
            if (!data) return;

            // Fill form inputs
            for (const [field, value] of Object.entries(data)) {
                const input = document.getElementById(field);
                if (input) {
                    input.value = value;
                    // Trigger change event if needed
                    input.dispatchEvent(new Event("change"));
                }
            }

            // Highlight the active button briefly
            presetButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            setTimeout(() => btn.classList.remove("active"), 1500);
        });
    });
}

/* ==========================================================================
   2. Prediction Form Handling (AJAX + Client Validation)
   ========================================================================== */
function initPredictionForm() {
    const form = document.getElementById("carPredictionForm");
    if (!form) return;

    const submitBtn = document.getElementById("submitBtn");
    const btnText = document.getElementById("btnText");
    const spinner = document.getElementById("btnSpinner");
    const errorBox = document.getElementById("clientErrorBox");
    const resultContainer = document.getElementById("resultContainer");

    form.addEventListener("submit", async (e) => {
        // We use AJAX for smooth user experience, with full fallback
        e.preventDefault();

        // Hide old error
        if (errorBox) {
            errorBox.style.display = "none";
            errorBox.textContent = "";
        }

        // Validate on client
        const formData = new FormData(form);
        const payload = {};
        formData.forEach((value, key) => {
            payload[key] = value;
        });

        // Quick validation checks
        if (Number(payload.km_driven) < 0) {
            showError("Kilometres driven cannot be negative.");
            return;
        }

        const year = Number(payload.year);
        if (year < 1990 || year > 2026) {
            showError("Please enter a manufacturing year between 1990 and 2026.");
            return;
        }

        // Show spinner
        setLoadingState(true);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
                showError(result.error || "Failed to calculate prediction.");
            } else {
                renderResult(result.data);
            }
        } catch (err) {
            console.error("AJAX error:", err);
            // In case of network glitch, submit standard form
            form.submit();
        } finally {
            setLoadingState(false);
        }
    });

    function setLoadingState(loading) {
        if (!submitBtn) return;
        submitBtn.disabled = loading;
        if (spinner) spinner.style.display = loading ? "inline-block" : "none";
        if (btnText) btnText.textContent = loading ? "Calculating Valuation..." : "Predict Selling Price";
    }

    function showError(msg) {
        if (errorBox) {
            errorBox.textContent = msg;
            errorBox.style.display = "flex";
            errorBox.scrollIntoView({ behavior: "smooth", block: "center" });
        } else {
            alert(msg);
        }
    }

    function renderResult(data) {
        if (!resultContainer) return;

        const summary = data.vehicle_summary;

        resultContainer.innerHTML = `
            <div class="card result-card">
                <div class="card-header" style="margin-bottom: 0.75rem;">
                    <div>
                        <span class="badge-tag">Valuation Result</span>
                        <div class="card-title" style="margin-top: 0.35rem;">
                            ${summary.brand} (${summary.year})
                        </div>
                    </div>
                    <span class="badge-tag" style="background: rgba(16, 185, 129, 0.2); color: #34d399;">
                        ✓ AI Model Verified
                    </span>
                </div>

                <div class="valuation-display">
                    <span class="valuation-tag">Estimated Resale Value</span>
                    <div class="valuation-amount">${data.predicted_price}</div>
                    <div class="valuation-words">${data.predicted_denomination}</div>
                </div>

                <div class="range-box">
                    <div class="range-label">
                        <span>Fair Market Confidence Band (±6%)</span>
                        <span>Estimated Range</span>
                    </div>
                    <div class="range-meter-track">
                        <div class="range-meter-fill"></div>
                    </div>
                    <div class="range-values">
                        <span>${data.confidence_low}</span>
                        <span>${data.confidence_high}</span>
                    </div>
                </div>

                <div style="font-size: 0.82rem; font-weight: 600; color: #34d399; margin-bottom: 0.5rem; text-transform: uppercase;">
                    Evaluated Specifications:
                </div>
                <div class="recap-grid">
                    <div class="recap-chip">
                        <span class="recap-chip-title">KM Driven</span>
                        <span class="recap-chip-value">${Number(summary.km_driven).toLocaleString('en-IN')} km</span>
                    </div>
                    <div class="recap-chip">
                        <span class="recap-chip-title">Fuel & Trans</span>
                        <span class="recap-chip-value">${summary.fuel} • ${summary.transmission}</span>
                    </div>
                    <div class="recap-chip">
                        <span class="recap-chip-title">Ownership</span>
                        <span class="recap-chip-value">${summary.owner}</span>
                    </div>
                    <div class="recap-chip">
                        <span class="recap-chip-title">Seller Type</span>
                        <span class="recap-chip-value">${summary.seller_type}</span>
                    </div>
                    <div class="recap-chip">
                        <span class="recap-chip-title">Engine & Power</span>
                        <span class="recap-chip-value">${summary.engine} CC • ${summary.max_power} bhp</span>
                    </div>
                    <div class="recap-chip">
                        <span class="recap-chip-title">Mileage & Seats</span>
                        <span class="recap-chip-value">${summary.mileage} kmpl • ${summary.seats} Seats</span>
                    </div>
                </div>

                <div class="disclaimer-box">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="8" x2="12" y2="12"></line>
                        <line x1="12" y1="16" x2="12.01" y2="16"></line>
                    </svg>
                    <div>
                        <strong>Academic Disclaimer:</strong> This estimated valuation is generated using a Random Forest machine learning model for educational and demonstration purposes. Real-world market values may fluctuate based on physical condition, service history, and local RTO taxes.
                    </div>
                </div>
            </div>
        `;

        resultContainer.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

/* ==========================================================================
   3. Dashboard Charts (Chart.js)
   ========================================================================== */
function initDashboardCharts() {
    const brandCanvas = document.getElementById("brandChart");
    const fuelCanvas = document.getElementById("fuelChart");
    const importanceCanvas = document.getElementById("importanceChart");
    const testSampleCanvas = document.getElementById("sampleComparisonChart");

    if (!brandCanvas && !fuelCanvas && !importanceCanvas && !testSampleCanvas) {
        return;
    }

    const Chart = typeof window !== "undefined" ? window.Chart : undefined;
    if (!Chart) {
        console.warn("Chart.js is not loaded yet or offline.");
        return;
    }

    // Default chart styling options for dark theme
    Chart.defaults.color = "#94a3b8";
    Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";

    // Fetch metrics from API
    fetch("/api/metrics")
        .then(res => res.json())
        .then(payload => {
            if (!payload.success) return;
            const metrics = payload.metrics || {};
            const summary = payload.summary || {};

            // 1. Feature Importance Chart
            if (importanceCanvas && metrics.feature_importances) {
                const labels = metrics.feature_importances.map(i => i.feature.toUpperCase());
                const values = metrics.feature_importances.map(i => (i.importance * 100).toFixed(1));

                new Chart(importanceCanvas, {
                    type: "bar",
                    data: {
                        labels: labels,
                        datasets: [{
                            label: "Relative Importance (%)",
                            data: values,
                            backgroundColor: "rgba(16, 185, 129, 0.75)",
                            borderColor: "#10b981",
                            borderWidth: 1,
                            borderRadius: 6
                        }]
                    },
                    options: {
                        indexAxis: "y",
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                callbacks: {
                                    label: ctx => ` Relative Weight: ${ctx.raw}%`
                                }
                            }
                        },
                        scales: {
                            x: {
                                grid: { color: "rgba(45, 90, 75, 0.15)" },
                                title: { display: true, text: "Importance (%)" }
                            },
                            y: {
                                grid: { display: false }
                            }
                        }
                    }
                });
            }

            // 2. Actual vs Predicted Sample Comparison Chart
            if (testSampleCanvas && metrics.test_samples) {
                const samples = metrics.test_samples.slice(0, 12);
                const labels = samples.map((s, idx) => `#${idx + 1} ${s.brand}`);
                const actuals = samples.map(s => (s.actual_price / 100000).toFixed(2));
                const preds = samples.map(s => (s.predicted_price / 100000).toFixed(2));

                new Chart(testSampleCanvas, {
                    type: "bar",
                    data: {
                        labels: labels,
                        datasets: [
                            {
                                label: "Actual Price (₹ Lakh)",
                                data: actuals,
                                backgroundColor: "rgba(52, 211, 153, 0.8)",
                                borderRadius: 4
                            },
                            {
                                label: "Predicted Price (₹ Lakh)",
                                data: preds,
                                backgroundColor: "rgba(132, 204, 22, 0.8)",
                                borderRadius: 4
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { position: "top" },
                            tooltip: {
                                callbacks: {
                                    label: ctx => ` ${ctx.dataset.label}: ₹ ${ctx.raw} Lakh`
                                }
                            }
                        },
                        scales: {
                            x: { grid: { display: false } },
                            y: {
                                grid: { color: "rgba(45, 90, 75, 0.15)" },
                                title: { display: true, text: "Price (₹ Lakh)" }
                            }
                        }
                    }
                });
            }

            // 3. Top Brands Market Share Chart
            if (brandCanvas) {
                const topBrandsObj = summary.top_brands || {
                    "Maruti": 593, "Hyundai": 559, "Toyota": 351, "Tata": 339,
                    "Mahindra": 306, "Honda": 283, "Kia": 153, "Renault": 136
                };
                const brandLabels = Object.keys(topBrandsObj);
                const brandCounts = Object.values(topBrandsObj);

                new Chart(brandCanvas, {
                    type: "bar",
                    data: {
                        labels: brandLabels,
                        datasets: [{
                            label: "Number of Vehicles",
                            data: brandCounts,
                            backgroundColor: "rgba(52, 211, 153, 0.75)",
                            borderColor: "#34d399",
                            borderWidth: 1,
                            borderRadius: 6
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { grid: { display: false } },
                            y: {
                                grid: { color: "rgba(45, 90, 75, 0.15)" },
                                title: { display: true, text: "Vehicle Count" }
                            }
                        }
                    }
                });
            }

            // 4. Fuel Distribution Doughnut Chart
            if (fuelCanvas) {
                const fuelObj = summary.fuel_counts || {
                    "Petrol": 1850, "Diesel": 1150, "CNG": 200
                };
                const fuelLabels = Object.keys(fuelObj);
                const fuelData = Object.values(fuelObj);

                new Chart(fuelCanvas, {
                    type: "doughnut",
                    data: {
                        labels: fuelLabels,
                        datasets: [{
                            data: fuelData,
                            backgroundColor: [
                                "rgba(16, 185, 129, 0.8)",
                                "rgba(132, 204, 22, 0.8)",
                                "rgba(56, 189, 248, 0.8)"
                            ],
                            borderColor: "#0f1e1a",
                            borderWidth: 3
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { position: "bottom" }
                        }
                    }
                });
            }
        })
        .catch(err => console.error("Error loading metrics for charts:", err));
}
