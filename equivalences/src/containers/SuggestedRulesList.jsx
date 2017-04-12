import React, { Component } from 'react';
import { bindActionCreators } from 'redux';
import { connect } from 'react-redux';
import { PropTypes } from 'prop-types';

import * as actions from 'redux/modules/list';
import List from 'components/List';

class SuggestedRulesList extends Component {
    static propTypes = {
        actions: PropTypes.shape(),
        list: PropTypes.arrayOf(PropTypes.object)
    }
    componentDidMount() {
        const { actions } = this.props;
        const { getList } = actions;
        getList();
    }
    render() {
        const { list, actions } = this.props;
        const { addRule } = actions;
        return (
            <List
                data={list}
                onClickAdd={(left, right) => function () {
                    addRule(left, right);
                }}
            />
        );
    }
}

function mapStateToProps(state) {
    return {
        list: state.list
    };
}

function mapDispatchToProps(dispatch) {
    return {
        actions: bindActionCreators(actions, dispatch)
    };
}

export default connect(mapStateToProps, mapDispatchToProps)(SuggestedRulesList);
