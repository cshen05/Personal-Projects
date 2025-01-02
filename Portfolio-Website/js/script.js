// Smooth scrolling for navigation links
document.querySelectorAll('nav a').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();

        // Scroll to the target section smoothly
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Intersection Observer for fade-in animations
const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in-visible'); // Add the fade-in-visible class
            observer.unobserve(entry.target); // Stop observing once the animation is applied
        }
    });
});

// Apply observer to elements with the fade-in class
document.querySelectorAll('.fade-in').forEach((el) => observer.observe(el));

// Parallax scrolling effect
document.addEventListener("scroll", () => {
    const sections = document.querySelectorAll("section");
    sections.forEach((section) => {
        const rect = section.getBoundingClientRect();
        if (rect.top >= 0 && rect.top <= window.innerHeight) {
            section.style.transform = `translateY(${rect.top / 10}px)`;
        }
    });
});