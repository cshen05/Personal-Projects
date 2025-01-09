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