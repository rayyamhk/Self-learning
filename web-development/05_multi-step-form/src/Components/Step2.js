import React from "react";

function Step2({info, setInfo, errorMsg}) {
    return(
        <form>
            {
                errorMsg !== "" && <div className="warning">{errorMsg}</div>
            }
            <div className="input-container">
                <label>gender</label>
                <div className="radio-container">
                    <input type="radio" name="gender" value="male" checked={info.gender==="male" ? "checked" : ""} onChange={(e) => {
                        setInfo(e);
                    }} />
                    <label>Male</label>
                </div>
                <div className="radio-container">
                    <input type="radio" name="gender" value="female" checked={info.gender==="female" ? "checked" : ""} onChange={(e) => {
                        setInfo(e);
                    }} />
                    <label>Female</label>
                </div>
            </div>
            <div className="input-container">
                <label>birth date</label>
                <input type="date" name="birth" value={info.birth} min="1920-1-1" max="2019-12-31" onChange={(e) => {
                        setInfo(e);
                }}></input>
            </div>
            <div className="input-container">
                <label>age</label>
                <input type="text" name="age" value={info.age} required onChange={(e) => {
                    setInfo(e);
                }}></input>
            </div>
            <div className="input-container">
                <label>phone number</label>
                <input type="text" name="phone" value={info.phone} required onChange={(e) => {
                    setInfo(e);
                }}></input>
            </div>
        </form>
    );
}

export default Step2;