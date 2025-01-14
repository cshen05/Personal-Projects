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

//Curtain Transition
document.addEventListener('DOMContentLoaded', () => {
    const curtain = document.querySelector('.curtain-overlay');
    const menuLinks = document.querySelectorAll('.nav-links a');

    menuLinks.forEach((link) => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetUrl = e.target.href;

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

            // Remove curtain classes after fade-out completes
            setTimeout(() => {
                curtain.classList.remove('curtain-drop', 'curtain-fade-out');
                console.log('Curtain reset.');
            }, 2500); // Match fade-out duration
        }, 1500); // Delay fade-out until after drop animation completes
    });
});