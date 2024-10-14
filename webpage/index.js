
let  locationCards = document.querySelector("location-card");
const card_img = document.createElement('img');
const stationListURL = "./station_locations.json"
TFL_LOGO_SRC = './images/tfl-logo.png';
card_img.src = TFL_LOGO_SRC;
card_img.height = 80;

let locations = []

const locationCardTemplate = document.querySelector("[data-location-template]")
const locationCardContainer = document.querySelector("[data-location-card-container]")
const searchInput = document.querySelector("[data-search]")

searchInput.addEventListener("input", e =>{
    const value = e.target.value.toLowerCase()
    locations.forEach(location =>{
        const isVisible = location.name.toLowerCase().includes(value)
        location.element.classList.toggle("hide",!isVisible)


    })
    console.log(value)
})

fetch(stationListURL).then(res => res.json()).then(data => {
        locations = data.map(location => {
            const card = locationCardTemplate.content.cloneNode(true).children[0];
            const locationName = card.querySelector("[data-location-name]")
            locationName.textContent = location.name
            new_img = card_img.cloneNode(true)
            card.appendChild(new_img)
            locationCardContainer.append(card)
            console.log(card)
            return {name: location.name, element: card}


        })
        

})

