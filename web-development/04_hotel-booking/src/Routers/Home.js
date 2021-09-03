import React from "react";

import Hero from "../Components/Hero";
import Banner from "../Components/Banner";
import Services from "../Components/Services";
import Title from "../Components/Title";
import FeaturedRooms from "../Components/FeaturedRooms";
import {Link} from "react-router-dom";

const Home = () => {
    return(
        <div>
            <Hero name="defaultHero">
                <Banner title="luxurious rooms" description="deluxe rooms starting at $299">
                    <Link className="btn-primary" to="/rooms">our rooms</Link>
                </Banner>
            </Hero>
            <Services>
                <Title title="Services"></Title>
            </Services>
            <FeaturedRooms>
                <Title title="Featured Rooms"></Title>
            </FeaturedRooms>
        </div>
    )
}

export default Home;