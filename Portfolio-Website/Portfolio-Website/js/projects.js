// Background Changer
window.addEventListener('scroll', () => {
    const overviewSection = document.querySelector('#overview'); // Page Overview
    const body = document.body; // Body element for overall background

    const overviewColor = [18, 48, 41]; // RGB for #123029
    const bodyColor = [26, 62, 52]; // RGB for #1a3e34

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

// Fade-in effect for projects
document.addEventListener('DOMContentLoaded', () => {
    const projects = document.querySelectorAll('.project');

    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = 1;
                    entry.target.style.transform = 'translateY(0)';
                    observer.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.1 } // Trigger when 10% of the project is visible
    );

    projects.forEach((project) => observer.observe(project));
});

window.addEventListener('scroll', () => {
    const header = document.querySelector('.sticky-header');
    if (window.scrollY > 50) {
        header.classList.add('active');
    } else {
        header.classList.remove('active');
    }
});

document.querySelector('.scroll-indicator').addEventListener('click', () => {
    const nextSection = document.querySelector('#projects');
    nextSection.scrollIntoView({ behavior: 'smooth' });
});

