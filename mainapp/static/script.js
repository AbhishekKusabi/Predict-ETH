'use strict';

// Function to GET csrftoken from Cookie
function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = jQuery.trim(cookies[i]);
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

var csrftoken = getCookie('csrftoken');

function csrfSafeMethod(method) {
    // these HTTP methods do not require CSRF protection
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}

// Function to set Request Header with `CSRFTOKEN`
function setRequestHeader(){
    $.ajaxSetup({
        beforeSend: function(xhr, settings) {
            if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
            }
        }
    });
}

function updateValue(){
    var data = {
        'date': document.getElementById("date").value,
    };
    setRequestHeader();

    $.ajax({
        dataType: 'json',
        type: 'POST',
        url: "/predict/",
        data: data,
        success: function (data) {
            console.log("Success:", data);
            document.getElementById("result_model").innerHTML = "$" + data.predicted_price;
            document.getElementById("result_high").innerHTML = "High: $" + data.predicted_high;             
            document.getElementById("result_low").innerHTML = "Low: $" + data.predicted_low;
        },
        error: function (jqXHR) {
            console.log("Error:", jqXHR);
            alert("updated failed!! ")
        }
    });
}

// Function to GET csrftoken from Cookie
function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = jQuery.trim(cookies[i]);
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

var csrftoken = getCookie('csrftoken');




function csrfSafeMethod(method) {
    // these HTTP methods do not require CSRF protection
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}

// Function to set Request Header with `CSRFTOKEN`
function setRequestHeader(){
    $.ajaxSetup({
        beforeSend: function(xhr, settings) {
            if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
            }
        }
    });
}

// 'use strict';

// // SELECTORS:
// const modal = document.querySelector('.modal');
// const overlay = document.querySelector('.overlay');
// const btnCloseModal = document.querySelector('.btn--close-modal');
// const btnsOpenModal = document.querySelectorAll('.btn--show-modal');
// const btnScrool = document.querySelector('.btn-hero');
// const btnContact = document.querySelector('.btn--contact');

// const section2 = document.querySelector('#section--2');
// const sectionCta = document.querySelector('.section-cta');

// const nav = document.querySelector('.nav');
// const header = document.querySelector('.header');
// const nav__link = document.querySelector('.nav__links');
// const nav__btn = document.querySelector('.nav__link--btn');
// const footerNav = document.querySelector('.footer__nav');
// const tabs = document.querySelectorAll('.vision__tab');
// const tabsContainer = document.querySelector('.vision__tab-container');
// const tabsContent = document.querySelectorAll('.vision__content');

// const countNumber = document.querySelectorAll('.number--heading');
// const numberObverve = document.querySelector('.number--obverve');

// const sendMailBtn = document.querySelector('ctf-btn');


// const openModal = function (e) {
//   e.preventDefault();
//   modal.classList.remove('hidden');
//   overlay.classList.remove('hidden');
// };

// const closeModal = function () {
//   modal.classList.add('hidden');
//   overlay.classList.add('hidden');
// };

// btnsOpenModal.forEach(btn => btn.addEventListener('click', openModal));
// btnCloseModal.addEventListener('click', closeModal);
// overlay.addEventListener('click', closeModal);
// document.addEventListener('keydown', function (e) {
//   if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
//     closeModal();
//   }
// });

// /////////////////////
// // Scroll
// ////////////////////
// btnScrool.addEventListener('click', function (e) {
//   section2.scrollIntoView({ behavior: 'smooth' });
// });
// btnContact.addEventListener('click', function (e) {
//   sectionCta.scrollIntoView({ behavior: 'smooth' });
// });

// // Page Navigation:
// nav__link.addEventListener('click', function (e) {
//   e.preventDefault();
//   const id = e.target.getAttribute('href');
//   if (!id) return;
//   if (e.target.classList.contains('nav__link--btn')) return;
//   document.getElementById(id).scrollIntoView({ behavior: 'smooth' });
// });

// //////////////////////
// // Footer navigation
// /////////////////////
// document.querySelector('.footer__nav').addEventListener('click', function (e) {
//   const id = e.target.getAttribute('href');
//   if (!id) return;
//   if (e.target.classList.contains('nav__link--btn')) return;
//   if (e.target.classList.contains('address')) return;
//   e.preventDefault();
//   document.getElementById(id).scrollIntoView({ behavior: 'smooth' });
// });
// ///////////////////////////
// // Nav hover handel
// //////////////////////////
// const handleHover = function (e, opacity) {
//   if (e.target.classList.contains('nav__link')) {
//     const link = e.target;
//     const sibling = link.closest('.nav').querySelectorAll('.nav__link');
//     sibling.forEach(el => {
//       if (el !== link) {
//         el.style.opacity = this;
//       }
//     });
//   }
// };
// nav.addEventListener('mouseover', handleHover.bind(0.5));
// nav.addEventListener('mouseout', handleHover.bind(1));

// ///////////////////
// //Sticky Nav
// //////////////////
// const navHeight = nav.getBoundingClientRect().height;
// const stickNav = function (entries) {
//   const [entry] = entries;
//   if (entry.isIntersecting) {
//     nav.classList.remove('sticky');
//     nav__link.style.color = '#fff';
//     nav__btn.style.backgroundColor = '#fff';
//     nav__btn.style.color = '#4b7de0';
//   } else {
//     nav.classList.add('sticky');
//     nav__link.style.color = '#555';
//     nav__btn.style.backgroundColor = '#4b7de0';
//     nav__btn.style.color = '#fff';
//   }
// };
// const headerObserver = new IntersectionObserver(stickNav, {
//   root: null,
//   threshold: 0,
//   rootMargin: '0px',
// });
// headerObserver.observe(header);
// ///////////////////
// // revelin section:
// ////////////////
// const allSections = document.querySelectorAll('.section');

// const revealSection = function (entries, observer) {
//   const [entry] = entries;
//   if (!entry.isIntersecting) return;
//   entry.target.classList.remove('section--hidden');
//   observer.unobserve(entry.target);
// };

// const sectionObserver = new IntersectionObserver(revealSection, {
//   root: null,
//   threshold: 0.15,
// });
// allSections.forEach(function (section) {
//   sectionObserver.observe(section);
//   section.classList.add('section--hidden');
// });

// // Tabbed Component
// tabsContainer.addEventListener('click', function (e) {
//   const clicked = e.target.closest('.vision__tab');
//   if (!clicked) return;
//   const dataTab = clicked.getAttribute('data-tab');
//   tabs.forEach(a => a.classList.remove('vision__tab--active'));
//   clicked.classList.add('vision__tab--active');

//   tabsContent.forEach(a => a.classList.remove('vision__content--active'));
//   document
//     .querySelector(`.vision__content--${dataTab}`)
//     .classList.add('vision__content--active');
// });

// // Slider:
// const slider = document.querySelector('.slider');
// const slides = document.querySelectorAll('.slide');
// const btnLeft = document.querySelector('.slider__btn--left');
// const btnRight = document.querySelector('.slider__btn--right');
// const dotContainer = document.querySelector('.dots');

// let currentSlide = 0;
// const maxSlides = slides.length;

// const goToSlide = function (slide) {
//   slides.forEach(
//     (s, i) => (s.style.transform = `translateX(${(i - slide) * 100}%)`)
//   );
// };
// goToSlide(0);

// const newSlide = function () {
//   if (currentSlide === maxSlides - 1) {
//     currentSlide = 0;
//   } else {
//     currentSlide++;
//   }
//   goToSlide(currentSlide);
//   activateDot(currentSlide);
// };

// const prevSlide = function () {
//   if (currentSlide === 0) {
//     currentSlide = maxSlides - 1;
//   } else {
//     currentSlide--;
//   }
//   goToSlide(currentSlide);
//   activateDot(currentSlide);
// };
// btnRight.addEventListener('click', newSlide);
// btnLeft.addEventListener('click', prevSlide);
// document.addEventListener('keydown', function (e) {
//   if (e.key === 'ArrowRight') newSlide();
//   else if (e.key === 'ArrowLeft') prevSlide();
//   activateDot(currentSlide);
// });

// const createDots = function () {
//   slides.forEach(function (_, i) {
//     dotContainer.insertAdjacentHTML(
//       'beforeend',
//       `<button class="dots__dot" data-slide="${i}"></button>`
//     );
//   });
// };
// createDots();
// const activateDot = function (slide) {
//   const dots = document.querySelectorAll('.dots__dot');

//   dots.forEach(function (s) {
//     s.style.backgroundColor = '#b9b9b9';
//     if (s.dataset.slide == slide) {
//       s.style.backgroundColor = '#5a8cef';
//     }
//   });
// };
// activateDot(0);
// dotContainer.addEventListener('click', function (e) {
//   if (e.target.classList.contains('dots__dot')) {
//     const { slide } = e.target.dataset;
//     goToSlide(slide);
//     activateDot(slide);
//   }
// });

// // /////////////////
// // EmailJs:
// ///////////////////
// function sendMail() {
//   var params = {
//     name: document.getElementById('name').value,
//     email: document.getElementById('email').value,
//     message: document.getElementById('message').value,
//   };
//   if (params.name === '' || params.email === '' || params.message === '') {
//     alert('Please fill the details properly');
//     return;
//   }

//   const serviceId = 'service_q9cchdh';
//   const tampleId = 'template_4rv707s';

//   emailjs
//     .send(serviceId, tampleId, params)
//     .then(res => {
//       document.getElementById('name').value = '';
//       document.getElementById('email').value = '';
//       document.getElementById('message').value = '';
//       console.log(res);
//       alert('Form submitted successfully!');
//     })
//     .catch(err => console.log(err));
// }

// document.getElementById('btn').addEventListener('click', function (e) {
//   e.preventDefault();
//   sendMail();
// });


