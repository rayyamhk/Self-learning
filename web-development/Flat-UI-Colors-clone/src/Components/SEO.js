import React from "react";
import {Helmet} from "react-helmet";

function SEO({title}) {
    return(
        <Helmet>
            <title>{title}</title>
        </Helmet>
    )
}

export default SEO;