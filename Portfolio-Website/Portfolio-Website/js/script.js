//Opening Functions
window.addEventListener('load', () => {
    const quoteOverlay = document.getElementById('quote-overlay');
    const heroSection = document.querySelector('.hero');
    const heroName = document.querySelector('.hero h1');
    const heroLinks = document.querySelector('.hero-links');
    const scrollIndicator = document.querySelector('.scroll-indicator');

    // Fade out and scale up the quote overlay after 3 seconds
    setTimeout(() => {
        quoteOverlay.style.opacity = '0'; // Fade out
        quoteOverlay.style.transform = 'scale(1.5)'; // Scale up
        quoteOverlay.style.transition = 'opacity 2s ease-in-out, transform 2s ease-in-out';

      // After the overlay fades out, reveal the hero section
        quoteOverlay.addEventListener('transitionend', (e) => {
            if (e.propertyName === 'opacity') {
                quoteOverlay.style.display = 'none';
                heroSection.classList.add('visible');
            }
            quoteOverlay.style.display = 'none'; // Hide the overlay
            heroSection.classList.add('visible'); // Make the hero section visible
    
            // Sequentially fade in elements within the hero section
            setTimeout(() => {
                heroName.style.opacity = '1';
                heroName.style.transform = 'translateY(0)';
                heroName.style.transition = 'opacity 1s ease-in-out, transform 1s ease-in-out';
            }, 500); // Delay for a smoother effect
    
            setTimeout(() => {
                heroLinks.style.opacity = '1';
                heroLinks.style.transform = 'translateY(0)';
                heroLinks.style.transition = 'opacity 1s ease-in-out, transform 1s ease-in-out';
            }, 1000);
    
            setTimeout(() => {
                scrollIndicator.style.opacity = '1';
                scrollIndicator.style.transform = 'translateY(0)';
                scrollIndicator.style.transition = 'opacity 1s ease-in-out, transform 1s ease-in-out';
            }, 1500);
        });
    }, 3000); // Delay before fading out the quote overlay
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

