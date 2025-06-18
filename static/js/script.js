document.addEventListener('DOMContentLoaded', () => {
    // Voice Command Functionality
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

    // Star Rating Functionality
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

    // Dropdown Menu Functionality
    const dropdowns = document.querySelectorAll('.dropdown');
    dropdowns.forEach(dropdown => {
        const button = dropdown.querySelector('.dropdown-button');
        const content = dropdown.querySelector('.dropdown-content');
        if (button && content) {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                const isActive = dropdown.classList.contains('active');
                document.querySelectorAll('.dropdown').forEach(d => d.classList.remove('active'));
                if (!isActive) {
                    dropdown.classList.add('active');
                    console.log(`Dropdown toggled: ${button.textContent}`);
                }
            });
            document.addEventListener('click', (e) => {
                if (!dropdown.contains(e.target)) {
                    dropdown.classList.remove('active');
                }
            });
        } else {
            console.warn('Dropdown button or content missing:', dropdown);
        }
    });

    // Ensure All Sections Are Visible
    const sections = document.querySelectorAll('.hero, .about, .features, .contact');
    sections.forEach(section => {
        section.style.display = 'block';
        console.log(`Section visible: ${section.className}`);
    });

    // Debug Navigation
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        console.log(`Nav link: ${link.textContent}, href: ${link.href}`);
    });
});