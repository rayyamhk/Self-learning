import React from "react";
import "./App.css";
import ProgressBar from "./Components/ProgressBar";
import Form from "./Components/Form";
import Buttons from "./Components/Buttons";
import End from "./Components/End";

//The 'this' auto-bind is available only inside the render and setState function

class App extends React.Component {
    state = {
        step: 1,
        total: 4,
        firstname: "",
        lastname: "",
        email: "",
        gender: "",
        birth: "",
        age: "",
        phone: "",
        message: "",
        source: "",
        accept: "",
        errorMsg: ""
    }

    render() {
        let info = {
            firstname: this.state.firstname,
            lastname: this.state.lastname,
            email: this.state.email,
            gender: this.state.gender,
            birth: this.state.birth,
            age: this.state.age,
            phone: this.state.phone,
            message: this.state.message,
            source: this.state.source,
            accept: this.state.accept
        }

        console.log(this.state.step);

        return(
            <div className="wrapper">
                <div className="container">
                    {
                        this.state.step <= this.state.total ? (
                            <>
                                <ProgressBar step={this.state.step} total={this.state.total}/>
                                <Form step={this.state.step} info={info} setInfo={this.setInfo} errorMsg={this.state.errorMsg}/>
                                <Buttons step={this.state.step} total={this.state.total} previous={this.previous} next={this.next} submit={this.submit}/> 
                            </>
                        ) : (
                            <End />
                        )
                        
                    }
                </div>
            </div>
        )
    }
    
    setInfo = (e) => {
        this.setState({[e.target.name] : e.target.value});
    }

    checkValidation = () => {
        const {firstname, lastname, email, gender, birth, age, phone, source, accept} = this.state;
        if(this.state.step === 1) {
            if(!firstname || !lastname || !email) {
                this.setState({ errorMsg : "Please fill in all blanks" })
                return false;
            }
            else {
                if(!email.match(/^.+@.+$/)) {
                    this.setState({ errorMsg : "Please input valid email format: email@example.com" })
                    return false;
                }
            }
        }
        else if(this.state.step === 2) {
            if(!gender || !birth || !age || !phone) {
                this.setState({ errorMsg : "Please fill in all blanks" })
                return false;
            }
            else {
                if(!age.match(/^[0-9]+$/)) {
                    this.setState({ errorMsg : "Please input valid age format" });
                    return false;
                }
                
                if(!phone.match(/^[0-9]+$/)) {
                    this.setState({ errorMsg : "Please input valid phone format: only contains digits" });
                    return false;
                }
            }
        }
        else if(this.state.step === 3) {
            if(!source || !accept) {
                this.setState({ errorMsg : "Please fill in all blanks"});
                return false;
            }
        }
        return true;
    }

    next = () => {
        if(this.state.step !== this.state.total) {
            if(this.checkValidation()) {
                this.setState({ errorMsg : "" });
                this.setState({ accept : "" });
                this.setState({ step : this.state.step + 1 });
            }
        }
    }

    previous = () => {
        if(this.state.step !== 1) {
            this.setState({ step : this.state.step - 1 });
            this.setState({ accept : "" });
            this.setState({ errorMsg : "" });
        }
    }

    submit = () => {
       if(this.state.step === this.state.total) {
           this.setState({ step : this.state.step + 1 });
       }
    }
}

export default App;