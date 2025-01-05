// sticky header
window.addEventListener('scroll', () => {
    const header = document.querySelector('.sticky-header');
    const hero = document.querySelector('.hero');
    const footer = document.querySelector('.footer');

    const heroHeight = hero.offsetHeight;
    const footerTop = footer.getBoundingClientRect().top;
    const viewportHeight = window.innerHeight;

    // Show sticky header after scrolling past the hero section
    if (window.scrollY > heroHeight) {
        header.classList.add('active');
    } else {
        header.classList.remove('active');
    }

    // Hide sticky header when footer is in view
    if (footerTop <= viewportHeight) {
        header.classList.add('hidden');
    } else {
        header.classList.remove('hidden');
    }
});

// Gradual Background Color Transition Between Sections
window.addEventListener('scroll', () => {
    const hero = document.querySelector('.hero');
    const about = document.querySelector('#about');
    const experience = document.querySelector('#experience');
    const scrollPosition = window.scrollY;

    // Define RGB colors for transitions
    const heroColor = [24, 39, 71]; // Navy (#182747)
    const aboutColor = [216, 216, 216]; // Light Gray (#D8D8D8)
    const experienceColor = [245, 245, 245]; // Muted Green (#647E68)

    const heroHeight = hero.offsetHeight;
    const aboutHeight = about.offsetHeight;

    // Transition from Hero to About
    if (scrollPosition <= heroHeight) {
        const ratio = scrollPosition / heroHeight; // Calculate scroll percentage
        const interpolatedColor = heroColor.map((start, index) =>
            Math.round(start + ratio * (aboutColor[index] - start))
        );
        hero.style.backgroundColor = `rgb(${interpolatedColor.join(',')})`;
        about.style.backgroundColor = `rgb(${aboutColor.join(',')})`;
    }

    // Transition from About to Experience
    const aboutTop = about.getBoundingClientRect().top;
    if (aboutTop <= window.innerHeight && aboutTop > -aboutHeight) {
        const ratio = Math.min(1, Math.abs(aboutTop / aboutHeight)); // Ensure ratio stays between 0 and 1
        const interpolatedColor = aboutColor.map((start, index) =>
            Math.round(start + ratio * (experienceColor[index] - start))
        );
        about.style.backgroundColor = `rgb(${interpolatedColor.join(',')})`;
        experience.style.backgroundColor = `rgb(${experienceColor.join(',')})`;
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

document.querySelector('.scroll-indicator').addEventListener('click', () => {
    const nextSection = document.querySelector('#about'); 
    nextSection.scrollIntoView({ behavior: 'smooth' });
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