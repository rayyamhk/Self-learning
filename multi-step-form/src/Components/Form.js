import React from "react";
import Step1 from "./Step1";
import Step2 from "./Step2";
import Step3 from "./Step3";
import Review from "./Review";

class Form extends React.Component {
    tag = [Step1, Step2, Step3, Review];

    render() {
        const {step, info, setInfo, errorMsg} = this.props;
        const TagName = this.tag[step - 1];
        return(
            <div className="form-container">
                <TagName info={info} setInfo={setInfo} errorMsg={errorMsg}/>
            </div>
        );
    }
}

export default Form;