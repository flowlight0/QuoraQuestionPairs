import React from 'react';
import { PropTypes } from 'prop-types';

export default function List({ data, onClickAdd }) {
    return (
        <div className="ts divided items">
            { data.map((val, key) => {
                return (
                    <div key={`item${key}`} className="item">
                        <div className="rule-header">
                            <div className="rule">{val.left.join(" ")} =&gt; {val.right.join(" ")}</div>
                            <div className="counts">
                                <div className="positive">{val.counts[0]}</div>
                                <div className="negative">{val.counts[1]}</div>
                            </div>
                            <input type="button" value="Add" onClick={onClickAdd(val.left, val.right)}/>
                            <input type="button" value="Add inverse" onClick={onClickAdd(val.right, val.left)}/>
                        </div>
                        <br/>
                        <div className="good-examples">
                            { val.good_examples.map((example, key) => {
                                    return (
                                        <div key={`good${key}`} className="example">
                                            <div className="q1">
                                                {example.q1_orig}
                                            </div>
                                            <div className="q2">
                                                {example.q2_orig}
                                            </div>
                                            <div className="q1_simp">
                                                {example.q1_simp}
                                            </div>
                                            <div className="q2_simp">
                                                {example.q2_simp}
                                            </div>
                                            <br/>
                                        </div>
                                    );
                            })}
                        </div>
                        <div className="bad-examples">
                            { val.bad_examples.map((example, key) => {
                                return (
                                    <div key={`good${key}`} className="example">
                                        <div className="q1">
                                            {example.q1_orig}
                                        </div>
                                        <div className="q2">
                                            {example.q2_orig}
                                        </div>
                                        <div className="q1_simp">
                                            {example.q1_simp}
                                        </div>
                                        <div className="q2_simp">
                                            {example.q2_simp}
                                        </div>
                                        <br/>
                                    </div>
                                );
                            })}
                        </div>
                        <hr/>
                    </div>

                );
            })}
        </div>
    );
}

List.defaultProps = {
    // data: [{
    //     id: 0,
    //     title: 'Title',
    //     link: '#',
    //     desc: 'Content',
    //     img: 'http://placehold.it/320x180',
    // }]
};

List.propTypes = {
    data: PropTypes.arrayOf(PropTypes.shape()),
    onClickAdd: PropTypes.func.isRequired
};
