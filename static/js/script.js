document.addEventListener('DOMContentLoaded', () => {
    const voiceButton = document.getElementById('voice-button');
    if (voiceButton) {
        voiceButton.addEventListener('click', () => {
            const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.onresult = (event) => {
                const command = event.results[0][0].transcript;
                fetch('/voice_command', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command })
                })
                .then(response => response.json())
                .then(data => alert(data.message))
                .catch(error => console.error('Voice command error:', error));
            };
            recognition.start();
        });
    }

    const stars = document.querySelectorAll('.rating .star');
    stars.forEach(star => {
        star.addEventListener('click', () => {
            const rating = star.dataset.value;
            const internshipId = star.closest('form').dataset.internshipId;
            document.querySelector(`form[data-internship-id="${internshipId}"] input[name="rating"]`).value = rating;
            stars.forEach(s => s.classList.remove('filled'));
            for (let i = 0; i < rating; i++) {
                stars[i].classList.add('filled');
            }
        });
    });
});