function FetchRoomByName(data, name) {
    for(let i = 0; i < data.length; i++) {
        if(data[i].fields.name === name) {
            return data[i];
        }
    }
    return null;
}

function FetchRoomByPath(data, path) {
    for(let i = 0; i < data.length; i++) {
        if(data[i].fields.slug === path) {
            return data[i];
        }
    }
    return null;
}

function FetchRoomByType(data, type) {
    if(type === "all") {
        return data;
    }
    else {
        let temp = []
        for(let i = 0; i < data.length; i++) {
            if(data[i].fields.type === type) {
                temp.push(data[i]);
            }
        }
        return temp;
    }
}

function FetchRoomByCapacity(data, guests) {
    let temp = [];
    for(let i = 0; i < data.length; i++) {
        if(data[i].fields.capacity >= guests) {
            temp.push(data[i]);
        }
    }
    return temp;
}

function FetchRoomByPrice(data, price) {
    let temp = [];
    for(let i = 0; i < data.length; i++) {
        if(data[i].fields.price <= price) {
            temp.push(data[i]);
        }
    }
    return temp;
}

function FetchRoomBySize(data, sizeLower, sizeUpper) {
    let temp = [];
    for(let i = 0; i < data.length; i++) {
        if(data[i].fields.size >= sizeLower && data[i].fields.size <= sizeUpper) {
            temp.push(data[i]);
        }
    }
    return temp;
}

function FetchRoomByBreakfast(data, breakfast) {
    if(breakfast) {
        let temp = [];
        for(let i = 0; i < data.length; i++) {
            if(data[i].fields.breakfast === breakfast) {
                temp.push(data[i]);
            }
        }
        return temp;
    }
    else {
        return data;
    }
}

function FetchRoomByPets(data, pets) {
    if(pets) {
        let temp = [];
        for(let i = 0; i < data.length; i++) {
            if(data[i].fields.pets === pets) {
                temp.push(data[i]);
            }
        }
        return temp;
    }
    else {
        return data;
    }
}

export {FetchRoomByName, FetchRoomByPath, FetchRoomByType, FetchRoomByCapacity, FetchRoomByPrice, FetchRoomBySize, FetchRoomByBreakfast, FetchRoomByPets}