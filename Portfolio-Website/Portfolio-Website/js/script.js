// Sticky Header Transition
window.addEventListener('scroll', () => {
    const header = document.querySelector('.sticky-header');
    const hero = document.querySelector('.hero');

    // Make sticky header visible after scrolling past the hero section
    if (window.scrollY > hero.offsetHeight - header.offsetHeight) {
        header.classList.add('active');
    } else {
        header.classList.remove('active');
    }
});

// Gradual Background Color Transition Between Sections
window.addEventListener('scroll', () => {
    const hero = document.querySelector('.hero');
    const about = document.querySelector('.about');
    const scrollPosition = window.scrollY;
    const heroHeight = hero.offsetHeight;

    if (scrollPosition <= heroHeight) {
        const ratio = scrollPosition / heroHeight; // Calculate scroll percentage
        const heroColor = [232, 220, 195]; // RGB for #E8DCC3 (Hero Section Color)
        const footerColor = [102, 66, 41]; // RGB for #664229 (Sticky Header/Footer Color)

        // Interpolate color based on scroll percentage
        const interpolatedColor = heroColor.map((start, index) =>
            Math.round(start + ratio * (footerColor[index] - start))
        );

        // Apply interpolated background color to hero section
        hero.style.backgroundColor = `rgb(${interpolatedColor.join(',')})`;
    }
});

// Smooth Scrolling for Navigation Links
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
            entry.target.classList.add('fade-in-visible'); // Add fade-in-visible class
            observer.unobserve(entry.target); // Stop observing after animation
        }
    });
});

// Apply observer to all elements with the fade-in class
document.querySelectorAll('.fade-in').forEach((el) => observer.observe(el));