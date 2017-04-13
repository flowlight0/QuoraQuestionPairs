/* global fetch */

export const GET = 'app/list/GET';
export const FAILED = 'app/list/FAILED';

export const ADD_RULE_REQUEST = 'app/list/ADD_RULE_REQUEST';

const reducer = (state = [], action) => {
    switch (action.type) {
    case GET:
        return action.data.map(item => ({
            left: item.left,
            right: item.right,
            counts: item.counts,
            good_examples: item.good_examples,
            bad_examples: item.bad_examples,
            // link: `/article/${item.id}`,
            // desc: item.body,
            // img: `http://lorempixel.com/320/180?t=${item.id}`,
        }));
    case FAILED:
        return [];
    default:
        return state;
    }
};

export default reducer;

export function getList(cb = () => {}) {
    return async dispatch => {
        try {
            const res = await fetch('/api/suggested');
            const json = await res.json();
            dispatch(gotList(json));
            cb();
        } catch (err) {
            dispatch(getListFailed());
            cb();
        }
    };
}

export function addRule(left, right, cb = () => {}) {
    return async dispatch => {
        try {
            const res = await fetch('/api/addrule', {
                method: "POST",
                body: JSON.stringify({
                    left: left,
                    right: right
                })
            });
            const json = await res.json();
            dispatch(gotList(json));
            cb();
        } catch (err) {
            dispatch(getListFailed());
            cb();
        }
    };
}

export function gotList(json) {
    return {
        type: GET,
        data: json,
    };
}

export function getListFailed() {
    return {
        type: FAILED,
    };
}
