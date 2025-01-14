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
    const header = document.querySelector('.sticky-header'); // Sticky header element
    const menuLinks = document.querySelectorAll('.nav-links a'); // All menu links

    if (!header) {
        console.error('Sticky header not found. Check your HTML structure.');
        return;
    }

    if (!menuLinks.length) {
        console.error('No menu links found. Ensure .nav-links a exists in your HTML.');
        return;
    }

    menuLinks.forEach((link) => {
        link.addEventListener('click', (e) => {
            e.preventDefault(); // Prevent default link behavior
            const targetUrl = e.target.href; // Get the URL of the target page

            // Add the curtain effect class
            header.classList.add('curtain-effect');

            // Wait for the curtain animation to fully cover the screen
            setTimeout(() => {
                // Navigate to the target page
                window.location.href = targetUrl;
            }, 750); // Halfway through the animation duration (1.5s)

            // Add an event listener for when the new page loads
            window.addEventListener('pageshow', () => {
                // Wait briefly, then remove the curtain effect
                setTimeout(() => {
                    header.classList.remove('curtain-effect');
                }, 500); // Adjust timing as needed for fade-out effect
            });
        });
    });
});