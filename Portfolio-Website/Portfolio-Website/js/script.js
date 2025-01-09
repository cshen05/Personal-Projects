//Opening Greeting
window.addEventListener('load', () => {
    const greetingOverlay = document.getElementById('greeting-overlay');
    const line1 = document.getElementById('greeting-line-1'); // Used for "Hey!" and "My name is Connor"
    const line3 = document.getElementById('greeting-line-3'); // Used for "Welcome to my Portfolio"

    // Typing animation helper function
    const typeText = (element, text, duration) => {
        element.textContent = ""; // Clear any existing text
        let i = 0;
        const interval = duration / text.length; // Calculate interval per character
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
        // Add a pause before fade-out
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
        setTimeout(() => typeText(line3, "Welcome to my Portfolio!", 2000), 1500); // Pause for 3 seconds after "My name is Connor"
    }, 10000); // Adjust timing for the sequence

    // Fade out greeting overlay
    setTimeout(() => {
        greetingOverlay.style.opacity = "0"; // Trigger fade-out
        greetingOverlay.style.transition = "opacity 2s ease-in-out";
    }, 15000); // Delay fade-out to match the longer animation time

    // Remove the overlay completely
    setTimeout(() => {
        greetingOverlay.style.display = "none"; // Hide the overlay
    }, 20000);
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

    const scrollPosition = window.scrollY;
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

// Smooth Scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();

        const target = document.querySelector(this.getAttribute('href'));
        target.scrollIntoView({
            behavior: 'smooth'
        });
    });
});

document.querySelector('.scroll-indicator').addEventListener('click', () => {
    const nextSection = document.querySelector('#about');
    nextSection.scrollIntoView({ behavior: 'smooth' });
});

// Fade-In and Zoom-In Animations on Scroll
document.addEventListener('DOMContentLoaded', () => {
    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    // Add fade-in and zoom-in classes dynamically
                    entry.target.classList.add('fade-in', 'zoom-in');
                    observer.unobserve(entry.target); // Stop observing once animation is applied
                }
            });
        },
        { threshold: 0.1 } // Trigger when 10% of the element is visible
    );
    // Apply observer to all elements with 'fade-in' or 'zoom-in' classes
    document.querySelectorAll('.fade-in, .zoom-in').forEach((el) => observer.observe(el));
});

document.addEventListener('DOMContentLoaded', () => {
    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                    observer.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.1 }
    );

    document.querySelectorAll('.fade-in').forEach((el) => observer.observe(el));
});

