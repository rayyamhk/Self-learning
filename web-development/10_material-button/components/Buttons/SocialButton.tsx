import React from "react";
import styled from "styled-components";
import * as FontAwesome from "react-icons/fa";
import { BoxShadowStyle, getBrandColor, RippleTrigger, halfTheSize } from "../../functions";
import Ripple from "../StyledComponents/Ripple";

interface Props {
    brand: string;
    href?: string;
    size?: string;
    margin?: string;
    border?: string;
    rippleColor?: string;
}

const IconButton:React.FC<Props> = ({ brand, href = "#", size = "40px", margin = "1rem", border = "2px", rippleColor = "#ffffff"}) => {
    const { backgroundColor, BrandTag } = getBrandColor(brand);
    const rippleRef = React.createRef() as React.RefObject<HTMLSpanElement>;
    return (
        <ButtonContainer href={href} background={backgroundColor} size={size} margin={margin} border={border} onClick={e => RippleTrigger(e, rippleRef.current as HTMLSpanElement)}>
            {React.createElement(FontAwesome[BrandTag])}
            <Ripple ref={rippleRef} color={rippleColor}/>
        </ButtonContainer>
    ) 
}

export default IconButton;

interface ButtonProps {
    background: string;
    size: string;
    margin: string;
    border: string;
}

const ButtonContainer = styled.a<ButtonProps>`
    text-decoration: none;
    color: white;
    font-size: ${props => halfTheSize(props.size)};
    background-color: ${props => props.background};
    box-shadow: ${props => BoxShadowStyle(false, props.background)};
    border-radius: ${props => props.border};
    width: ${props => props.size};
    height: ${props => props.size};
    margin: ${props => props.margin};
    position: relative;
    display: inline-flex;
    justify-content: center;
    align-items: center;
    transition: ${props => props.theme.primaryTransition};
    overflow: hidden;

    &:hover,
    &:focus {
        box-shadow: ${props => BoxShadowStyle(true, props.background)};
    }
`