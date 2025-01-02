// Transition Hero into Sticky Navigation
window.addEventListener('scroll', () => {
    const header = document.querySelector('.sticky-header');
    const hero = document.querySelector('.hero');

    if (window.scrollY > hero.offsetHeight - header.offsetHeight) {
        header.classList.add('active');
    } else {
        header.classList.remove('active');
    }
});

// Gradual Color Transition Between Sections
window.addEventListener('scroll', () => {
    const hero = document.querySelector('.hero');
    const about = document.querySelector('.about');
    const scrollPosition = window.scrollY;
    const heroHeight = hero.offsetHeight;

    if (scrollPosition <= heroHeight) {
        const ratio = scrollPosition / heroHeight; // Calculate scroll percentage
        const heroColor = [250, 249, 246]; // RGB for #FAF9F6 (Hero Section Color)
        const aboutColor = [210, 180, 140]; // RGB for #D2B48C (About Section Color)

        // Calculate interpolated color
        const interpolatedColor = heroColor.map((start, index) =>
            Math.round(start + ratio * (aboutColor[index] - start))
        );

        // Apply interpolated color to hero section background
        hero.style.backgroundColor = `rgb(${interpolatedColor.join(',')})`;
    }
});

// Smooth Scrolling for Links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();

        const target = document.querySelector(this.getAttribute('href'));
        target.scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Fade-In Animations on Scroll
const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in-visible'); // Add the fade-in-visible class
            observer.unobserve(entry.target); // Stop observing once animation is applied
        }
    });
});

// Apply observer to elements with the fade-in class
document.querySelectorAll('.fade-in').forEach((el) => observer.observe(el));