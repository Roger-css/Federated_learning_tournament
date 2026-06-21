/**
 * Federated Learning Dashboard - HUD Application Logic (Cyber-Glass)
 */

const activeCharts = {};

document.addEventListener('DOMContentLoaded', () => {
    const pageType = document.body.id;
    const pageInitializers = {
        'dashboard-page': initDashboardPage,
        'monitoring-page': initMonitoringPage,
        'detection-page': initDetectionPage,
        'comparison-page': initComparisonPage
    };

    if (pageInitializers[pageType]) {
        pageInitializers[pageType]();
    }
});

function initDashboardPage() {
    createChart('f1Chart', 'line', {
        labels: ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7'],
        datasets: [{
            label: 'Global F1 Accuracy',
            data: [0.65, 0.72, 0.78, 0.83, 0.87, 0.90, 0.92],
            borderColor: '#0ea5e9',
            backgroundColor: (ctx) => {
                const gradient = ctx.chart.ctx.createLinearGradient(0, 0, 0, 400);
                gradient.addColorStop(0, 'rgba(14, 165, 233, 0.2)');
                gradient.addColorStop(1, 'rgba(14, 165, 233, 0)');
                return gradient;
            },
            fill: true,
            tension: 0.5,
            borderWidth: 4,
            pointBackgroundColor: '#0ea5e9',
            pointBorderColor: '#fff',
            pointRadius: 6,
            pointHoverRadius: 8
        }]
    });
}

function initMonitoringPage() {
    createChart('sensorTimeChart', 'line', {
        labels: Array.from({length: 20}, (_, i) => i + ":00"),
        datasets: [{
            label: 'PSI_TELEMETRY',
            data: Array.from({length: 20}, () => Math.random() * 10 + 50),
            borderColor: '#10b981',
            backgroundColor: 'transparent',
            borderWidth: 3,
            pointRadius: 0,
            tension: 0.1
        }]
    });
}

function initDetectionPage() {
    createChart('confidenceGauge', 'doughnut', {
        labels: ['SCAN_MATCH', 'SCAN_REMAIN'],
        datasets: [{
            data: [94, 6],
            backgroundColor: ['#ec4899', 'rgba(15, 23, 42, 0.05)'],
            borderWidth: 0,
            circumference: 180,
            rotation: 270
        }]
    }, { 
        cutout: '85%',
        plugins: { legend: { display: false } }
    });
}

function initComparisonPage() {
    createChart('comparisonChart', 'bar', {
        labels: ['NODE_A', 'NODE_B', 'NODE_G', 'NODE_D'],
        datasets: [
            {
                label: 'STNDALONE_UNIT',
                data: [0.72, 0.65, 0.78, 0.68],
                backgroundColor: 'rgba(99, 102, 241, 0.7)',
                borderRadius: 5
            },
            {
                label: 'FEDERATED_CORE',
                data: [0.88, 0.85, 0.82, 0.89],
                backgroundColor: 'rgba(14, 165, 233, 0.7)',
                borderRadius: 5
            }
        ]
    });
}

function createChart(canvasId, type, chartData, customOptions = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const context = canvas.getContext('2d');
    if (activeCharts[canvasId]) activeCharts[canvasId].destroy();

    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { 
            legend: { 
                display: customOptions.plugins?.legend?.display !== false,
                position: 'top',
                labels: { 
                    color: '#64748b',
                    font: { family: 'JetBrains Mono', size: 10, weight: '700' },
                    usePointStyle: true,
                    padding: 20
                } 
            } 
        },
        scales: (type === 'line' || type === 'bar') ? {
            y: { 
                grid: { color: 'rgba(15, 23, 42, 0.06)' }, 
                ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 9 } },
                border: { display: false }
            },
            x: { 
                grid: { display: false }, 
                ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 9 } },
                border: { display: false }
            }
        } : {}
    };

    activeCharts[canvasId] = new Chart(context, {
        type: type,
        data: chartData,
        options: { ...defaultOptions, ...customOptions }
    });
}
