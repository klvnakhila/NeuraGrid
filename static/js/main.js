// Main JavaScript file for NeuraGrid

document.addEventListener('DOMContentLoaded', function() {
    // Initialize any interactive elements
    initializeApp();
});

function initializeApp() {
    // Add any global initialization code here
    console.log('NeuraGrid application initialized');
    
    // Initialize forecast duration buttons if they exist
    const forecastButtons = document.querySelectorAll('.forecast-btn');
    if (forecastButtons.length > 0) {
        forecastButtons.forEach(button => {
            button.addEventListener('click', handleForecastSelection);
        });
    }
}

function handleForecastSelection(event) {
    // Remove active class from all buttons
    document.querySelectorAll('.forecast-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Add active class to clicked button
    event.target.classList.add('active');
    
    // Get selected duration
    const duration = event.target.dataset.duration;
    
    // Update forecast display
    updateForecastDisplay(duration);
}

function updateForecastDisplay(duration) {
    // This function will be implemented to update the forecast display
    // based on the selected duration (1-day or 7-day)
    console.log(`Updating forecast display for ${duration}`);
    
    // You can add AJAX calls here to fetch new forecast data
    // or update the existing display
}

// Utility functions
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function formatDate(date) {
    return new Date(date).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}
