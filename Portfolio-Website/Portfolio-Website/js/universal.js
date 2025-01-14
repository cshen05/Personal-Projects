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
    const header = document.querySelector('.sticky-header');
    const menuLinks = document.querySelectorAll('.nav-links a');

    menuLinks.forEach((link) => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetUrl = e.target.href;

            // Add the curtain effect class
            header.classList.add('curtain-effect');
            console.log('Curtain effect class added');

            // Navigate to the new page after the curtain fully drops
            setTimeout(() => {
                window.location.href = targetUrl; // Navigate to the target page
            }, 1500); // Match curtain-drop duration
        });
    });

    // Ensure the curtain stays until the new page is fully loaded
    window.addEventListener('pageshow', () => {
        const header = document.querySelector('.sticky-header');
        console.log('New page fully loaded. Starting fade-out.');

        // Start the fade-out animation
        setTimeout(() => {
            header.classList.remove('curtain-effect');
            console.log('Curtain effect class removed');
        }, 2500); // Fade-out duration
    });
});