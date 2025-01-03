// Sticky navigation bar behavior
window.addEventListener('scroll', () => {
    const header = document.querySelector('.sticky-header');
    const footer = document.querySelector('.footer');
  
    const footerTop = footer.getBoundingClientRect().top;
    const viewportHeight = window.innerHeight;
  
    // Hide navigation when footer comes into view
    if (footerTop <= viewportHeight) {
      header.classList.add('hidden');
    } else {
      header.classList.remove('hidden');
    }
  });

// Gradual Background Color Transition Between Sections
window.addEventListener('scroll', () => {
    const hero = document.querySelector('.hero');
    const about = document.querySelector('.about');
    const scrollPosition = window.scrollY;
    const heroHeight = hero.offsetHeight;

    // Define RGB colors for transitions
    const heroColor = [24, 39, 71]; // Navy (#182747)
    const aboutColor = [104, 126, 104]; // Muted Green (#647E68)

    // Ensure transition occurs smoothly within the hero height
    if (scrollPosition <= heroHeight) {
        const ratio = scrollPosition / heroHeight; // Calculate scroll percentage
        const interpolatedColor = heroColor.map((start, index) =>
            Math.round(start + ratio * (aboutColor[index] - start))
        );

        // Apply interpolated background color to hero section
        hero.style.backgroundColor = `rgb(${interpolatedColor.join(',')})`;
        about.style.backgroundColor = `rgb(${interpolatedColor.join(',')})`;
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