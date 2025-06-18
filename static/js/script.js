document.addEventListener('DOMContentLoaded', () => {
    // Debug: Log DOM readiness
    console.log('DOM fully loaded');

    // Voice Command Functionality
    const voiceButton = document.getElementById('voice-button');
    if (voiceButton) {
        voiceButton.addEventListener('click', () => {
            const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.onresult = (event) => {
                const command = event.results[0][0].transcript;
                console.log('Voice command:', command);
                fetch('/voice_command', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command })
                })
                .then(response => response.json())
                .then(data => {
                    console.log('Voice command response:', data);
                    alert(data.message || data.error);
                })
                .catch(error => console.error('Voice command error:', error));
            };
            recognition.start();
        });
    } else {
        console.warn('Voice button not found');
    }

    // Star Rating Functionality
    const stars = document.querySelectorAll('.rating .star');
    if (stars.length > 0) {
        stars.forEach(star => {
            star.addEventListener('click', () => {
                const rating = star.dataset.value;
                const internshipId = star.closest('form').dataset.internshipId;
                console.log(`Rating ${rating} for internship ${internshipId}`);
                document.querySelector(`form[data-internship-id="${internshipId}"] input[name="rating"]`).value = rating;
                stars.forEach(s => s.classList.remove('filled'));
                for (let i = 0; i < rating; i++) {
                    stars[i].classList.add('filled');
                }
            });
        });
    } else {
        console.warn('No rating stars found');
    }

    // Dropdown Menu Functionality
    const dropdowns = document.querySelectorAll('.dropdown');
    console.log(`Found ${dropdowns.length} dropdowns`);
    dropdowns.forEach(dropdown => {
        const button = dropdown.querySelector('.dropdown-button');
        const content = dropdown.querySelector('.dropdown-content');
        if (button && content) {
            console.log(`Dropdown button: ${button.textContent}, content children: ${content.children.length}`);
            // Click event for touch devices
            button.addEventListener('click', (e) => {
                e.preventDefault();
                const isActive = dropdown.classList.contains('active');
                document.querySelectorAll('.dropdown').forEach(d => d.classList.remove('active'));
                if (!isActive) {
                    dropdown.classList.add('active');
                    console.log(`Dropdown toggled: ${button.textContent}`);
                }
            });
            // Close on click outside
            document.addEventListener('click', (e) => {
                if (!dropdown.contains(e.target)) {
                    dropdown.classList.remove('active');
                }
            });
            // Hover for desktop
            dropdown.addEventListener('mouseenter', () => {
                dropdown.classList.add('active');
                console.log(`Dropdown hovered: ${button.textContent}`);
            });
            dropdown.addEventListener('mouseleave', () => {
                dropdown.classList.remove('active');
            });
        } else {
            console.error('Dropdown missing button or content:', dropdown);
        }
    });

    // Ensure All Sections Are Visible
    const sections = document.querySelectorAll('.hero, .about, .features, .contact');
    console.log(`Found ${sections.length} sections`);
    sections.forEach(section => {
        section.style.display = 'block';
        section.style.visibility = 'visible';
        console.log(`Section visible: ${section.className}`);
    });

    // Debug Navigation Links
    const navLinks = document.querySelectorAll('.nav-link');
    console.log(`Found ${navLinks.length} nav links`);
    navLinks.forEach(link => {
        console.log(`Nav link: ${link.textContent}, href: ${link.href}`);
    });

    // Debug Dropdown Content
    const dropdownContents = document.querySelectorAll('.dropdown-content a');
    console.log(`Found ${dropdownContents.length} dropdown options`);
    dropdownContents.forEach(option => {
        console.log(`Dropdown option: ${option.textContent}, href: ${option.href}`);
    });
});