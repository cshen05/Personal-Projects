//Opening Functions
window.addEventListener('load', () => {
    const greetingOverlay = document.getElementById('greeting-overlay');
    const line1 = document.getElementById('greeting-line-1');
    const line2 = document.getElementById('greeting-line-2');

    // Type "Hey!"
    setTimeout(() => {
        line1.textContent = "Hey!";
        line1.style.animation = "typing 2s steps(3, end), blink-caret 0.5s step-end infinite";
    }, 500);

    // Replace "Hey!" with "My name is Connor"
    setTimeout(() => {
        line1.style.animation = "none"; // Remove typing animation for fade-out
        line1.textContent = ""; // Clear line1
        setTimeout(() => {
            line1.textContent = "My name is Connor";
            line1.style.animation = "typing 3s steps(17, end), blink-caret 0.5s step-end";
        }, 200);
    }, 3000);

    // Type "Welcome to my Portfolio Website!"
    setTimeout(() => {
        line2.textContent = "Welcome to my Portfolio Website!";
        line2.style.animation = "typing 3s steps(32, end), blink-caret 0.5s step-end";
    }, 7000);

    // Fade out the greeting overlay
    setTimeout(() => {
        greetingOverlay.style.opacity = "0";
        greetingOverlay.style.transition = "opacity 1s ease-in-out";
    }, 12000);

    // Remove the overlay completely
    setTimeout(() => {
        greetingOverlay.style.display = "none";
    }, 13000);
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

