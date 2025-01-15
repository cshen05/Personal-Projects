//Opening Greeting
window.addEventListener('load', () => {
    const greetingOverlay = document.getElementById('greeting-overlay');
    // const greetingShown = localStorage.getItem('greetingShown')
    const sessionVisited = sessionStorage.getItem('sessionVisited'); // Specific to this session

    if (!sessionVisited) {
        // Mark greeting as shown
        sessionStorage.setItem('sessionVisited', 'true');

        greetingOverlay.style.display = 'flex'; // Ensure it's visible

        const line1 = document.getElementById('greeting-line-1'); // Used for "Hey!" and "My name is Connor"
        const line3 = document.getElementById('greeting-line-3'); // Used for "Welcome to my Portfolio"

        // skip greeting
        window.addEventListener('click', () => {
            greetingOverlay.style.display = "none";
        })

        // Typing animation helper function
        const typeText = (element, text, duration) => {
            element.textContent = ""; // Clear any existing text
            let i = 0;
            const interval = Math.min(duration / text.length, 75); // Calculate interval per character
            const typing = setInterval(() => {
                if (i < text.length) {
                    element.textContent += text[i];
                    i++;
                } else {
                    clearInterval(typing); // Stop the typing animation
                }
            }, interval);
        };

        // Typing sequence with pause and fade-out
        setTimeout(() => typeText(line1, "Hey!", 1000), 500); // Type "Hey!" in 1 second
        setTimeout(() => {
            setTimeout(() => {
                line1.classList.add('fade-out'); // Apply fade-out class
                setTimeout(() => {
                    line1.classList.remove('fade-out'); // Remove fade-out class
                    line1.style.opacity = "1"; // Reset opacity for new text
                    typeText(line1, "My name is Connor", 1500); // Type "My name is Connor" in the same position
                }, 3000); // Wait for fade-out to complete before typing next line
            }, 1000); // Pause for 1 second after typing "Hey!"
        }, 1000); // Delay before the pause and fade-out logic

        setTimeout(() => {
            setTimeout(() => typeText(line3, "Welcome to my Website!", 2000));
        }, 8000); // Adjust timing for the sequence

        // Fade out greeting overlay
        setTimeout(() => {
            greetingOverlay.style.opacity = "0"; // Trigger fade-out
            greetingOverlay.style.transition = "opacity 2s ease-in-out";
        }, 11000); // Delay fade-out to match the longer animation time

        // Remove the overlay completely
        setTimeout(() => {
            greetingOverlay.style.display = "none"; // Hide the overlay
        }, 12500);
    } else {
        greetingOverlay.style.display = 'none'
    }
});

//Background Changer
window.addEventListener('scroll', () => {
    const sections = [
        document.querySelector('.hero'),
        document.querySelector('#about'),
        document.querySelector('#experience'),
    ];
    const sectionColors = [
        [24, 39, 71],   // Hero: Navy (#182747)
        [216, 216, 216], // About: Light Gray (#D8D8D8)
        [245, 245, 245], // Experience: Very Light Gray (#F5F5F5)
    ];

    const viewportHeight = window.innerHeight;

    sections.forEach((section, index) => {
        const nextSection = sections[index + 1];
        if (!nextSection) return; // Skip if no next section

        const sectionTop = section.getBoundingClientRect().top;
        const nextSectionTop = nextSection.getBoundingClientRect().top;

        if (sectionTop <= viewportHeight && nextSectionTop > 0) {
            const ratio = Math.min(1, Math.max(0, 1 - nextSectionTop / viewportHeight));
            const interpolatedColor = sectionColors[index].map((start, i) =>
                Math.round(start + ratio * (sectionColors[index + 1][i] - start))
            );

            section.style.backgroundColor = `rgb(${interpolatedColor.join(',')})`;
            nextSection.style.backgroundColor = `rgb(${sectionColors[index + 1].join(',')})`;
        }
    });
});

// Sticky Header
window.addEventListener('scroll', () => {
    const header = document.querySelector('.sticky-header');
    const hero = document.querySelector('.hero');
    const footer = document.querySelector('.footer');

    const heroHeight = hero.offsetHeight;
    const footerTop = footer.getBoundingClientRect().top;
    const viewportHeight = window.innerHeight;

    // Show sticky header after scrolling past the hero section
    if (window.scrollY > heroHeight) {
        header.classList.add('active');
    } else {
        header.classList.remove('active');
    }

    // Hide sticky header when footer is in view
    if (footerTop <= viewportHeight) {
        header.classList.add('hidden');
    } else {
        header.classList.remove('hidden');
    }
});

document.querySelector('.scroll-indicator').addEventListener('click', () => {
    const nextSection = document.querySelector('#about');
    nextSection.scrollIntoView({ behavior: 'smooth' });
});

//Curtain Transition
document.addEventListener('DOMContentLoaded', () => {
    const curtain = document.querySelector('.curtain-overlay');
    const menuLinks = document.querySelectorAll('.nav-links a');
    const body = document.body;

    menuLinks.forEach((link) => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetUrl = e.target.href;

            // Block interactions
            body.classList.add('no-pointer-events');

            // Trigger the curtain drop
            curtain.classList.add('curtain-drop');
            console.log('Curtain dropped.');

            // Navigate to the new page after the curtain fully drops
            setTimeout(() => {
                window.location.href = targetUrl;
            }, 1500); // Match the drop duration
        });
    });

    // On the new page, fade out the curtain after it loads
    window.addEventListener('pageshow', () => {
        console.log('New page fully loaded. Preparing fade-out.');

        // Start fade-out after ensuring the drop has completed
        setTimeout(() => {
            curtain.classList.add('curtain-fade-out');
            console.log('Curtain fading out.');

            // Allow interactions after fade-out completes
            setTimeout(() => {
                curtain.classList.remove('curtain-drop', 'curtain-fade-out');
                body.classList.remove('no-pointer-events'); // Re-enable interactions
                console.log('Curtain reset and interactions enabled.');
            }, 2500); // Match fade-out duration
        }, 1500); // Delay fade-out until after drop animation completes
    });
});