// Background Changer for Resume Page
window.addEventListener('scroll', () => {
    const overviewSection = document.querySelector('#overview'); // Page Overview
    const body = document.body; // Body element for overall background

    const overviewColor = [255, 204, 128]; // RGB for #ffcc80 (Light Orange)
    const bodyColor = [230, 81, 0]; // RGB for #e65100 (Bold Orange)

    const viewportHeight = window.innerHeight;
    const sectionTop = overviewSection.getBoundingClientRect().top;
    const sectionBottom = overviewSection.getBoundingClientRect().bottom;

    // Calculate scroll ratio within the overview section
    let ratio = Math.min(1, Math.max(0, 1 - sectionBottom / viewportHeight));

    // Interpolate between the two colors based on the scroll ratio
    const interpolatedColor = overviewColor.map((start, i) =>
        Math.round(start + ratio * (bodyColor[i] - start))
    );

    // Apply the interpolated color to the body background
    body.style.backgroundColor = `rgb(${interpolatedColor.join(',')})`;
});

document.addEventListener('DOMContentLoaded', () => {
    const pdfViewer = document.querySelector('.resume-viewer');

    // Observer for fade-in effect
    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    pdfViewer.classList.add('visible'); // Add the visible class
                    observer.unobserve(pdfViewer); // Stop observing after it fades in
                }
            });
        },
        { threshold: 0.1 } // Trigger when 10% of the viewer is visible
    );

    observer.observe(pdfViewer);
});

// Sticky Header Activation
window.addEventListener('scroll', () => {
    const header = document.querySelector('.sticky-header');
    if (window.scrollY > 50) {
        header.classList.add('active');
    } else {
        header.classList.remove('active');
    }
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