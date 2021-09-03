import React from "react";
import ReactDom from "react-dom";
import {BrowserRouter as Router, Route, Switch} from "react-router-dom";
import "./App.css";
import Home from "./Routers/Home";
import Rooms from "./Routers/Rooms";
import SingleRoom from "./Routers/SingleRoom";
import Error from "./Routers/Error";

import NavBar from "./Components/NavBar";


class App extends React.Component {
    render() {
        return(
            <>
                <Router basename={process.env.PUBLIC_URL}>
                    <NavBar />
                    {/* Switch will render the first-matched page */}
                    {/* Route with no path will always be matched (ERROR) */}
                    <Switch>
                        <Route exact path="/" component={Home} />
                        <Route exact path="/rooms">
                            <Rooms/>
                        </Route>
                        <Route exact path="/rooms/:path" component={SingleRoom} />
                        <Route component={Error} />
                    </Switch>
                </Router>
            </>
        );
    }
}

ReactDom.render(
    <App />,
    document.getElementById("root")
);