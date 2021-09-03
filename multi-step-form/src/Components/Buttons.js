import React from "react";
import Button from "./Button";

function Buttons({step, total, previous, next, submit}) {
    return(
        <div className="buttons-container">
            {
                step === 1 ? (
                    <Button text="next" extra={next}/>
                ) : step === total ? (
                    <>
                    <Button text="submit" extra={submit} />
                    <Button text="previous" extra={previous}/>
                    </>
                ) : (
                    <>
                    <Button text="next" extra ={next}/>
                    <Button text="previous" extra={previous}/>
                    </>
                )
            }
        </div>
    );
}

export default Buttons;