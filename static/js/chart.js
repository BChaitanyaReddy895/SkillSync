document.addEventListener('DOMContentLoaded', () => {
    const ctx = document.getElementById('analyticsChart');
    if (ctx) {
        const data = JSON.parse(ctx.dataset.analytics);
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.map(item => item.name),
                datasets: [{
                    label: 'Acceptance Likelihood (%)',
                    data: data.map(item => item.acceptance_prob),
                    backgroundColor: '#00796b'
                }]
            },
            options: {
                scales: {
                    y: { beginAtZero: true, max: 100 }
                }
            }
        });
    }
});