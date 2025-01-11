// Background Changer
window.addEventListener('scroll', () => {
    const sections = [
        document.querySelector('#overview'), // Overview Section
        ...document.querySelectorAll('#projects'), // Each project section
    ];
    const sectionColors = [
        [18, 48, 41],
        [31, 92, 74]
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