import React from "react";
import {BrowserRouter as Router, Route, Switch} from "react-router-dom";
import ColorsPage from "./Routers/ColorsPage";
import LandingPage from "./Routers/LandingPage";

class App extends React.Component{
    render() {
        return(
            <Router basename={process.env.PUBLIC_URL}>
                <Switch>
                    <Route exact path="/" component={LandingPage}/>
                    <Route exact path="/palette/:id" component={ColorsPage}/>
                </Switch>
            </Router>
            
        )
    }
}

export default App;