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

let currentFactIndex = 0;

const funFacts = [
  "Huge Arsenal FC supporter, UT Football is an obligation, and sometimes the Houston Dynamo",
  "My go-to comfort game is Clash of Clans",
  "I was originally a swimmer :) Was my high school's varsity captain too",
  "I can read Korean but I don't understand what I'm reading",
  "My end goal in life is to travel the world",
  "I can cook 2-minute instant noodles in 1.5 minutes (Personal Record)"
];

// Toggle banner open/close
document.querySelector('.fun-facts-icon').addEventListener('click', () => {
  const banner = document.querySelector('.fun-facts-banner');
  banner.classList.toggle('open'); // Toggle the banner open/close
});

// Flip the card to show the next fun fact
document.querySelector('.fun-facts-card').addEventListener('click', () => {
  const card = document.querySelector('.fun-facts-card');
  const funFactText = document.getElementById('fun-fact');

  card.classList.add('flip'); // Add flip animation

  setTimeout(() => {
    currentFactIndex = (currentFactIndex + 1) % funFacts.length;
    funFactText.textContent = funFacts[currentFactIndex];
    card.classList.remove('flip'); // Reset flip for the next click
  }, 600); // Match the CSS transition duration
});

// Auto-close the banner when scrolling out of the section
window.addEventListener('scroll', () => {
  const banner = document.querySelector('.fun-facts-banner');
  const section = document.querySelector('#about'); // Replace with the section containing the fun facts
  const sectionTop = section.getBoundingClientRect().top;
  const sectionBottom = section.getBoundingClientRect().bottom;

  // Check if the section is out of view
  if (sectionBottom < 0 || sectionTop > window.innerHeight) {
    banner.classList.remove('open'); // Close the banner
  }
});