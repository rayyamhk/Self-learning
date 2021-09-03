import React from "react";
import Hero from "../Components/Hero";
import Banner from "../Components/Banner";
import {Link} from "react-router-dom";

const Error = () => {
    return(
        <div>
            <Hero name="defaultHero">
                <Banner title="404" description="page not found">
                    <Link className="btn-primary" to="/">return home</Link>
                </Banner>
            </Hero>
        </div>
    )
}

export default Error